import json
import argparse
import random

import torch
from torch import nn
from torch.nn import functional as F
import torchvision
import os
from PIL import Image
import matplotlib.pyplot as plt
import wandb

from fourier_layer import FourierLayer


def factorize_with_smallest_difference(n):
    factors = []

    for i in range(1, int(n ** 0.5) + 1):
        if n % i == 0:
            factors.append((i, n // i))

    best_factors = min(factors, key=lambda pair: abs(pair[0] - pair[1]))

    return best_factors


parser = argparse.ArgumentParser()
parser.add_argument('--path', type=str, default='views', help="path to .json file containing the image info")
parser.add_argument('--train', action='store_true', help="train a new model")
parser.add_argument('--eval', action='store_true', help="evaluate currently saved model")
parser.add_argument('--resume', action='store_true', help="continue training currently saved model")
parser.add_argument('--cuda', action='store_true', help="use cuda for training")
parser.add_argument('--verbose', action='store_true', help="print epoch to console")
parser.add_argument('--wandb', type=str, default=None, help="wandb project to log this run to")
args = parser.parse_args()

assert args.train or args.eval or args.resume, "You must provide one of --train, --eval or --resume"
assert sum(
    map(int, [args.train, args.eval, args.resume])) == 1, "You must provide only one of --train, --eval or --resume"

transform = torchvision.transforms.Compose([
    torchvision.transforms.ToTensor(),
])

# load hyperparams from file
with open('3d/hyperparameters.json') as f:
    hyperparams_dict = json.load(f)
    hyperparams = argparse.Namespace(**hyperparams_dict)

if args.wandb is not None:
    wandb.login(anonymous='allow')
    if args.train or args.resume:
        wandb.init(
            project=args.wandb,
            config=hyperparams_dict
        )
    else:
        wandb.init(
            project=args.wandb
        )

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

if hyperparams.batch_size == -1:
    hyperparams.batch_size = pics.shape[0]
assert pics.shape[
           0] % hyperparams.batch_size == 0, f"Cannot divide {pics.shape[0]} images evenly into {hyperparams.batch_size} groups"
assert pics.shape[
           1] % hyperparams.n_image_splits == 0, f"Cannot horizontally divide images of horizontal size {pics.shape[1]} evenly into {hyperparams.n_image_splits} equal segments"

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

assert model_in.shape[:-1] == pics.shape[:-1], f"Model input and expected output do not align in shape. {model_in.shape[:-1]=}; {pics.shape[:-1]=}"

# randomly permute the data if program was launched in train or resume mode
if args.train or args.resume:
    for dim in range(model_in.dim() - 1):
        size = model_in.size(dim)
        permutation = torch.randperm(size)
        model_in = model_in.index_select(dim, permutation)
        pics = pics.index_select(dim, permutation)

# split model_in into pieces because it requires 600GiB GPU RAM otherwise D:
X = []
Y = []
x_batch_splits = model_in.split(hyperparams.batch_size)
y_batch_splits = pics.split(hyperparams.batch_size)

for x_batch_split, y_batch_split in zip(x_batch_splits, y_batch_splits):
    X.extend(x_batch_split.split(x_batch_split.shape[1] // hyperparams.n_image_splits, dim=1))
    Y.extend(y_batch_split.split(y_batch_split.shape[1] // hyperparams.n_image_splits, dim=1))
    del x_batch_split, y_batch_split

# define the model
if args.train:
    fourier_layer = FourierLayer(
        hyperparams.n_fourier_features,
        hyperparams.frequency_spacing,
        fan_in=model_in.shape[-1],
        no_grad=True
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
optim = torch.optim.AdamW(model.parameters(), lr=0)
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
    losses = []
    n_epochs = 0
    best_checkpoint_loss = float('inf')
    loss = float('NaN')
    print("Starting/resuming training...")
    for epoch_group_idx, lr_with_epochs in enumerate(hyperparams.learning_rates):
        losses.append([])
        for i in range(len(optim.param_groups)):
            optim.param_groups[i]['lr'] = lr_with_epochs['lr']
        for e in range(lr_with_epochs['epochs']):
            losses[-1].append([])
            seed = random.getrandbits(64)
            random.seed(seed)
            random.shuffle(X)
            random.seed(seed)
            random.shuffle(Y)
            for i, x, y in zip(range(len(X)), X, Y):
                pred = (model(x) + 1) / 2
                L = F.mse_loss(pred, y)
                optim.zero_grad()
                L.backward()
                optim.step()
                losses[-1][-1].append(L.item())
            loss = sum(losses[-1][-1]) / len(losses[-1][-1])
            if args.wandb is not None:
                wandb.log({f"{'train' if args.train else 'resume'}/loss": loss,
                           f"{'train' if args.train else 'resume'}/lr": lr_with_epochs['lr']})
            if args.verbose:
                print(f'Epoch {n_epochs}   Loss {loss}')
            n_epochs += 1
            # todo: do this for both train and val loss (save best train loss model and best val loss model and compare)
            if lr_with_epochs['checkpoints'] > 0:  # if you want to checkpoint at all
                if lr_with_epochs['checkpoints'] > lr_with_epochs['epochs']:
                    raise Exception(f"In epoch group {epoch_group_idx}, checkpoints (={lr_with_epochs['checkpoints']}) was set to something higher than epochs (={lr_with_epochs['epochs']}). You cannot make more checkpoints than there are epochs!")
                else:
                    is_checkpoint_epoch = (e + 1) % (lr_with_epochs['epochs'] // lr_with_epochs['checkpoints']) == 0
                if is_checkpoint_epoch and loss <= best_checkpoint_loss:  # want to checkpoint and loss is better than at the end of last checkpointed epoch group (an epoch group is a group of epochs with shared lr)
                    best_checkpoint_loss = loss
                    torch.save(model, '3d/model.ckpt')
                    print("Saved the model to model.ckpt")
        print("Training finished. Saving model if it's better than last checkpoint.")
        if loss <= best_checkpoint_loss:  # want to checkpoint and loss is better than at the end of last checkpointed epoch group (an epoch group is a group of epochs with shared lr)
            best_checkpoint_loss = loss
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
    if args.wandb is not None:
        wandb_pred = [wandb.Image(img.to('cpu').detach().numpy()) for img in imgs]
        wandb_ground_truth = [wandb.Image(x.to('cpu').detach().numpy()) for x in X]
        wandb_eval_table = wandb.Table(columns=["prediction", "ground_truth"], data=zip(wandb_pred, wandb_ground_truth))
        wandb.log({"eval/results": wandb_eval_table})

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

# in case this code is copied into a notebook
if args.wandb is not None:
    wandb.finish()
