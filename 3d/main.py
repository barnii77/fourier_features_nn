import json
import argparse
import warnings
import torch
import torch.nn as nn
from torch.nn import functional as F
import torchvision
import os
from PIL import Image
import matplotlib.pyplot as plt


parser = argparse.ArgumentParser()
parser.add_argument('--path', type=str, default='data/', help="path to .json file containing the image info")
parser.add_argument('--train', action='store_true', help="train a new model")
parser.add_argument('--eval', action='store_true', help="evaluate currently saved model")
parser.add_argument('--resume', action='store_true', help="continue training currently saved model")
parser.add_argument('--cuda', action='store_true', help="use cuda for training")
parser.add_argument('--batch-size', type=int, default=1, help="batch size")
parser.add_argument('--n-image-splits', type=int, default=1)
parser.add_argument('--print-epoch', action='store_true', help="print epoch to console")
args = parser.parse_args()

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

assert json_fn is not None, "Could not find a json file containing "

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
assert pics.shape[0] % args.batch_size == 0, f"Cannot divide {pics.shape[0]} images evenly into {args.batch_size} groups"
assert pics.shape[1] % args.n_image_splits == 0, f"Cannot horizontally divide images of horizontal size {pics.shape[1]} evenly into {args.n_image_splits} equal segments"


# define fourier input module
class FourierLayer(nn.Module):
    def __init__(self, fan_in: int, n_fourier_features: int, freq_spacing: str = 'logarithmic'):
        super().__init__()
        self.nff = n_fourier_features
        self.out_dim = 2 * fan_in * self.nff
        self.freq_spacing = freq_spacing

    def forward(self, x):
        x = x.view(x.shape + (1,))
        if self.freq_spacing == 'logarithmic':
            feature_mul = 2 ** torch.arange(0, self.nff, device=x.device)
        elif self.freq_spacing == 'linear':
            feature_mul = torch.arange(1, self.nff + 1, device=x.device)
        else:
            raise Exception(f'Unknown frequency spacing paradigm: {self.freq_spacing}')
        sines = torch.sin(x * feature_mul)
        cosines = torch.cos(x * feature_mul)
        # sines.shape = cosines.shape
        return torch.concat([sines.view(sines.shape + (1,)), cosines.view(cosines.shape + (1,))], dim=-1).flatten(-3, -1)


# load hyperparams from file
with open('3d/hyperparameters.json') as f:
    hyperparams = argparse.Namespace(**json.load(f))

# define model input
# pure pixel position input + reshape
model_in = torch.concat([torch.arange(pics.shape[1]).unsqueeze(-1).repeat_interleave(pics.shape[2], dim=1).unsqueeze(-1), torch.arange(pics.shape[2]).unsqueeze(0).repeat_interleave(pics.shape[1], dim=0).unsqueeze(-1)], dim=-1) / torch.tensor(list(pics.shape[1:])[:-1]) * 2 - 1
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
    fourier_layer = FourierLayer(model_in.shape[-1], hyperparams.n_fourier_features, hyperparams.frequency_spacing)
    model = nn.Sequential(
        fourier_layer,
        nn.Linear(fourier_layer.out_dim, hyperparams.n_hidden),
        nn.ReLU(),
        nn.Linear(hyperparams.n_hidden, hyperparams.n_hidden),
        nn.ReLU(),
        nn.Linear(hyperparams.n_hidden, 3),
        nn.Tanh()
    )
elif args.eval or args.resume:
    model = torch.load('3d/model.ckpt')
else:
    raise Exception("You must provide one of --train, --eval or --resume")

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
    for e in range(hyperparams.epochs):
        for i, x, y in zip(range(len(X)), X, Y):
            pred = (model(x) + 1) / 2
            loss = F.mse_loss(pred, y)
            optim.zero_grad()
            loss.backward()
            optim.step()
            if i % 100 == 0:
                print(f'Sample {i}')
        if args.print_epoch:
            print(f'Epoch {e}')
else:
    warnings.warn("Not fully implemented feature. Will only eval on one hardcoded part of the dataset.")
    pred = (model(X[0]) + 1) / 2

if args.train:
    torch.save(model, '3d/model.ckpt')

# show the result
if hyperparams.epochs > 0:
    plt.imshow(pred.detach().to('cpu').permute((1, 0, 2)).numpy())
    plt.show()

# todo: inefficiencies >> loading images all into one tensor, then concat, then split, then convert to fourier representation. Could instead do concat on every single tensor and split etc individually and after one another to avoid having lots of temporary memory allocation at once
