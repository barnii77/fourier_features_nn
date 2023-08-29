import json
import random
import torch
from torch import nn
from torch.nn import functional as F

from fourier_layer import FourierLayer


# implement this class yourself for your model!
class Model(nn.Module):
    def __init__(self, fan_in: int, fan_out: int):
        super(Model, self).__init__()
        with open('3d/hyperparameters.json') as f:
            hyperparams = json.load(f)['model.py']  # only load the hyperparameters that are meant to be used in this file
        self.fan_in = fan_in
        self.fan_out = fan_out
        self.fourier_layer = FourierLayer(hyperparams['n_fourier_features'], fan_in=fan_in)
        self.act = nn.LeakyReLU(inplace=True)
        self.fc1 = nn.Linear(self.fourier_layer.out_dim, hyperparams['n_hidden'])
        self.fc2 = nn.Linear(hyperparams['n_hidden'], hyperparams['n_hidden'])
        self.fc3 = nn.Linear(hyperparams['n_hidden'], fan_out)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.fourier_layer(x)

        # model here
        # -------
        x = self.fc1(x)
        x = self.act(x)
        x = self.fc2(x)
        x = self.act(x)
        x = self.fc3(x)
        # -------
        x = F.sigmoid(2 * x)  # 2 * x to make sigmoid steeper for gradients
        return x


if __name__ == '__main__':
    model = Model(random.randint(1, 100), random.randint(1, 100))
    for _ in range(1000):
        model_in = torch.randn(model.fan_in)
        model_out = model(model_in)
        assert torch.all((0 <= model_out) * (model_out <= 1)).item(), f"Model failed output range test (0 <= output <= 1 for each output).\nModel input:\n{model_in}\n\nModel output:\n{model_out}"
