import json
import argparse
import torch
import torch.nn as nn
from torch.nn import functional as F
import torchvision
import os
from PIL import Image
import matplotlib.pyplot as plt

from fourier_layer import FourierLayer


def factorize_with_smallest_difference(n):
    factors = []

    for i in range(1, int(n ** 0.5) + 1):
        if n % i == 0:
            factors.append((i, n // i))

    best_factors = min(factors, key=lambda pair: abs(pair[0] - pair[1]))

    return best_factors


parser = argparse.ArgumentParser()
parser.add_argument('--path', type=str, default='data/', help="path to .json file containing the image info")
parser.add_argument('--train', action='store_true', help="train a new model")
parser.add_argument('--eval', action='store_true', help="evaluate currently saved model")
parser.add_argument('--resume', action='store_true', help="continue training currently saved model")
parser.add_argument('--cuda', action='store_true', help="use cuda for training")
parser.add_argument('--batch-size', type=int, default=1, help="batch size")
parser.add_argument('--n-image-splits', type=int, default=1)
parser.add_argument('--verbose', action='store_true', help="print epoch to console")
args = parser.parse_args()

assert args.train or args.eval or args.resume, "You must provide one of --train, --eval or --resume"
assert sum(
    map(int, [args.train, args.eval, args.resume])) == 1, "You must provide only one of --train, --eval or --resume"

transform = torchvision.transforms.Compose([
    torchvision.transforms.ToTensor(),
])

# load images
pics, cam_pos, cam_rot = [], [], []
json_fn = None
for fn in os.listdir(f"3d/{args.path}/"):
    if fn.endswith('.json'):
        assert json_fn is None, "More than one json file in data directory. Remove all but the necessary one."
        json_fn = fn

assert json_fn is not None, "Could not find a json file containing dataset info"

with open(f"3d/{args.path}/{json_fn}") as f:
    data_json = json.load(f)

for x in data_json:
    pic = Image.open("3d/" + x["image"]).convert('RGB')
    pic = transform(pic).permute((2, 1, 0)).unsqueeze(0)
    pics.append(pic)
    pos = torch.tensor(list(map(float, x["position"].split(', '))))
    cam_pos.append(pos)
    rot = torch.tensor(list(map(float, x["rotation"].split(', '))))
    cam_rot.append(rot)

assert all([i.shape == j.shape for i, j in zip(pics, pics[1:])]), "Not all images have the same shape."
pics = torch.concat(pics)
cam_pos = torch.stack(cam_pos)
cam_rot = torch.stack(cam_rot)

if args.batch_size == -1:
    args.batch_size = pics.shape[0]
assert pics.shape[
           0] % args.batch_size == 0, f"Cannot divide {pics.shape[0]} images evenly into {args.batch_size} groups"
assert pics.shape[
           1] % args.n_image_splits == 0, f"Cannot horizontally divide images of horizontal size {pics.shape[1]} evenly into {args.n_image_splits} equal segments"

# load hyperparams from file
with open('3d/hyperparameters.json') as f:
    hyperparams = argparse.Namespace(**json.load(f))

# define model input
# pure pixel position input + reshape
'''
model_in = torch.concat(
    [torch.arange(pics.shape[1]).unsqueeze(-1).repeat_interleave(pics.shape[2], dim=1).unsqueeze(-1),
     torch.arange(pics.shape[2]).unsqueeze(0).repeat_interleave(pics.shape[1], dim=0).unsqueeze(-1)],
    dim=-1) / torch.tensor(list(pics.shape[1:-1])) * 2 - 1

Does the same thing (but worse) as:
'''
model_in = torch.stack(
    [
        torch.linspace(-1, 1, pics.shape[1]).unsqueeze(-1).repeat_interleave(pics.shape[2], dim=1),
        torch.linspace(-1, 1, pics.shape[2]).unsqueeze(0).repeat_interleave(pics.shape[1], dim=0)
    ],
    dim=-1
)
model_in = model_in.unsqueeze(0).repeat_interleave(pics.shape[0], dim=0)
# reshape camera position and rotation
cam_pos = (cam_pos.view((cam_pos.shape[0], 1, 1, cam_pos.shape[1]))
           .repeat_interleave(model_in.shape[1], dim=1)
           .repeat_interleave(model_in.shape[2], dim=2))
cam_rot = (cam_rot.view((cam_rot.shape[0], 1, 1, cam_rot.shape[1]))
           .repeat_interleave(model_in.shape[1], dim=1)
           .repeat_interleave(model_in.shape[2], dim=2))
# concat
model_in = torch.concat([cam_pos, cam_rot, model_in], dim=-1)

# split model_in into pieces because it requires 600GiB GPU RAM otherwise D:
X = []
Y = []
x_batch_splits = model_in.split(args.batch_size)
y_batch_splits = pics.split(args.batch_size)

for x_batch_split, y_batch_split in zip(x_batch_splits, y_batch_splits):
    X.extend(x_batch_split.split(x_batch_split.shape[1] // args.n_image_splits, dim=1))
    Y.extend(y_batch_split.split(y_batch_split.shape[1] // args.n_image_splits, dim=1))
    del x_batch_split, y_batch_split

# define the model
if args.train:
    fourier_layer = FourierLayer(
        hyperparams.n_fourier_features,
        hyperparams.frequency_spacing,
        fan_in=model_in.shape[-1]
    )
    model = nn.Sequential(
        fourier_layer,
        nn.Linear(fourier_layer.out_dim, hyperparams.n_hidden),
        nn.ReLU(),
        nn.Linear(hyperparams.n_hidden, hyperparams.n_hidden),
        nn.ReLU(),
        nn.Linear(hyperparams.n_hidden, 3),
        nn.Tanh()
    )
else:
    model = torch.load('3d/model.ckpt')

# define the optimizer
optim = torch.optim.AdamW(model.parameters(), hyperparams.lr)
pred = None

# move to gpu
if args.cuda:
    if torch.cuda.is_available():
        # model_in = model_in.to('cuda')  not used anymore!!!
        for i, x in enumerate(X):
            X[i] = x.to('cuda')
        for i, y in enumerate(Y):
            Y[i] = y.to('cuda')
        model = model.to('cuda')
        # pics = pics.to('cuda')  not used anymore!!!
        print('Moved parameters/IO to cuda')
    else:
        raise Exception('Cuda not available -> Failed to move parameters/IO to cuda')

# delete unnecessary variables for gpu memory
del parser, transform, pics, cam_pos, cam_rot, model_in, x_batch_splits, y_batch_splits

# main training loop
if args.train or args.resume:
    print("Starting/resuming training...")
    for e in range(hyperparams.epochs):
        for i, x, y in zip(range(len(X)), X, Y):
            pred = (model(x) + 1) / 2
            loss = F.mse_loss(pred, y)
            optim.zero_grad()
            loss.backward()
            optim.step()
        if args.verbose:
            print(f'Epoch {e}/{hyperparams.epochs}')

    torch.save(model, '3d/model.ckpt')
    print("Saved the model to model.ckpt")
else:
    print("Starting evaluation...")
    pred = []
    for i, x in enumerate(X):
        if args.verbose:
            print(f"Eval sample {i}/{len(X)}")
        pred.append(((model(x) + 1) / 2).to('cpu').detach())
    pred = torch.concat(pred)
    print("Generating output plots...")
    # detach and split images (or parts of images if --n-image-splits is bigger than 1)
    imgs = pred.transpose(2, 1).numpy()
    imgs = [imgs[i] for i in range(imgs.shape[0])]

    num_images = len(imgs)
    num_rows, num_cols = factorize_with_smallest_difference(num_images)
    fig, axes = plt.subplots(num_rows, num_cols, figsize=(5 * num_cols, 5))

    # Plot each image in its respective subplot
    for i in range(num_images):
        if num_images > 1:
            ax = axes[i]
        else:
            ax = axes  # Handle the case when there's only one image

        ax.imshow(imgs[i])
        ax.set_title(f'Image(/part) {i + 1}')

    # Adjust layout to prevent overlapping titles and labels
    plt.tight_layout()
    print("Showing plots...")
    plt.show()
