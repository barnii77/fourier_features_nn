import json
import argparse
import math

import torch
import torch.nn as nn
from torch.nn import functional as F
import torchvision
# import numpy as np
from PIL import Image
import matplotlib.pyplot as plt


parser = argparse.ArgumentParser()
parser.add_argument('--learn', type=str, default='2d/image.jpg')
parser.add_argument('--train', action='store_true')
parser.add_argument('--eval', action='store_true')
parser.add_argument('--resume', action='store_true')
parser.add_argument('--cuda', action='store_true')
parser.add_argument('--no-fourier', action='store_true')
args = parser.parse_args()

# load image
pic = Image.open(args.learn).convert('RGB')
# pic = torch.from_numpy(np.array(pic.getdata(), dtype=np.float32) / 255).view((pic.width, pic.height, 3))
transform = torchvision.transforms.Compose([
    torchvision.transforms.ToTensor(),
])

pic = transform(pic).permute((2, 1, 0))


# define fourier input module
class FourierLayer(nn.Module):
    def __init__(self, fan_in: int, n_fourier_features: int, freq_spacing: str = 'logarithmic', trainable_frequencies: bool = False, frequency_bias: bool = False):
        super().__init__()
        self.nff = n_fourier_features
        self.out_dim = 2 * fan_in * self.nff
        self.freq_spacing = freq_spacing
        if self.freq_spacing == 'logarithmic':
            self.frequencies = math.pi * 2 ** torch.arange(1, self.nff + 1)  # = 2 * math.pi * 2 ** torch.arange(0, self.nff)
        elif self.freq_spacing == 'linear':
            self.frequencies = 2 * math.pi * torch.arange(1, self.nff + 1)
        else:
            raise Exception(f'Unknown frequency spacing paradigm: {self.freq_spacing}')
        if trainable_frequencies:
            if frequency_bias:
                self.frequency_bias = nn.Parameter(torch.zeros_like(self.frequencies))
            self.frequencies = nn.Parameter(self.frequencies, requires_grad=True)
        self.trainable_frequencies = trainable_frequencies
        self.has_frequency_bias = frequency_bias

    def forward(self, x):
        x = x.view(x.shape + (1,))
        frequencies = self.frequencies.to(x.device)
        fin = x * frequencies
        if self.has_frequency_bias:
            fin += self.frequency_bias
        sines = torch.sin(fin)
        cosines = torch.cos(fin)
        # sines.shape = cosines.shape
        return torch.concat([sines.view(sines.shape + (1,)), cosines.view(cosines.shape + (1,))], dim=-1).flatten(-3, -1)


# load hyperparams from file
with open('2d/hyperparameters.json') as f:
    hyperparams = argparse.Namespace(**json.load(f))

# define model input
model_in = torch.concat([torch.arange(pic.shape[0]).unsqueeze(-1).repeat_interleave(pic.shape[1], dim=1).unsqueeze(-1), torch.arange(pic.shape[1]).unsqueeze(0).repeat_interleave(pic.shape[0], dim=0).unsqueeze(-1)], dim=-1) / torch.tensor(list(pic.shape)[:-1]) * 2 - 1

# define the model
if args.train:
    if args.no_fourier:
        model = nn.Sequential(
            nn.Linear(2, hyperparams.n_hidden),
            nn.ReLU(),
            nn.Linear(hyperparams.n_hidden, hyperparams.n_hidden),
            nn.ReLU(),
            nn.Linear(hyperparams.n_hidden, 3),
            nn.Tanh()
        )
    else:
        fourier_layer = FourierLayer(model_in.shape[-1], hyperparams.n_fourier_features, hyperparams.frequency_spacing, True, True)
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
    model = torch.load('2d/model.ckpt')
else:
    raise Exception("You must provide one of --train, --eval or --resume")

# define the optimizer
optim = torch.optim.AdamW(model.parameters(), hyperparams.lr)
pred = None

# move to gpu
if args.cuda:
    if torch.cuda.is_available():
        model_in = model_in.to('cuda')
        model = model.to('cuda')
        pic = pic.to('cuda')
        print('Moved parameters/IO to cuda')
    else:
        raise Exception('Failed to move parameters/IO to cuda')

# main training loop
if args.train or args.resume:
    for e in range(hyperparams.epochs):
        pred = (model(model_in) + 1) / 2
        loss = F.mse_loss(pred, pic)
        optim.zero_grad()
        loss.backward()
        optim.step()
        if hyperparams.log_epoch_to_console:
            print(f'Epoch {e}')
else:
    pred = (model(model_in) + 1) / 2

if args.train:
    torch.save(model, '2d/model.ckpt')

# show the result
if hyperparams.epochs > 0:
    plt.imshow(pred.detach().to('cpu').permute((1, 0, 2)).numpy())
    plt.show()
