import json
import random
import torch
from torch import nn
from torch.nn import functional as F

from fourier_layer import FourierLayer

with open('hyperparameters.json') as f:
    hyperparams = json.load(f)['model.py']  # only load the hyperparameters that are meant to be used in this file


# implement this class yourself for your model!
class Model(nn.Module):
    def __init__(self, fan_in: int, fan_out: int):
        super(Model, self).__init__()
        self.fan_in = fan_in
        self.fan_out = fan_out
        self.fourier_layer = FourierLayer(hyperparams['n_fourier_features'], fan_in=fan_in)
        self.fc1 = nn.Linear(self.fourier_layer.out_dim, fan_out)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.fourier_layer(x)
        x = self.fc1(x)
        x = F.sigmoid(x + x)  # 2 * x to make sigmoid steeper for gradients (and x + x because (technically) 5x more performant than 2 * x)
        return x


if __name__ == '__main__':
    model = Model(random.randint(1, 100), random.randint(1, 100))
    for _ in range(1000):
        model_in = torch.randn(model.fan_in)
        model_out = model(model_in)
        assert torch.all((0 <= model_out) * (model_out <= 1)).item(), f"Model failed output range test (0 <= output <= 1 for each output).\nModel input:\n{model_in}\n\nModel output:\n{model_out}"
