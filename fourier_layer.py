import math
import torch
import torch.nn as nn


class FourierLayer(nn.Module):
    """
    A torch.nn.Module child class that maps every input feature to n_fourier_features sines and n_fourier_features cosines with different (either linearly or exponentially growing) frequencies and concatenates the ouptuts.
    n_fourier_features is the number of pairs of sine/cosine outputs that every input feature is mapped to
    freq_spacing_type must be either 'logarithmic' or 'linear' and controls how the sine/cosine frequencies are spaced ('logarithmic' is recommended and typically performs a lot better)
    fan_in is an optional argument controlling how many inputs the layer will get. if provided, the FourierLayer.out_dim will be an integer storing the output size. otherwise, it will be None
    no_grad controls whether the input is allowed to have requires_grad = True. no_grad = True -> not allowed
    """
    def __init__(self, n_fourier_features: int, freq_spacing_type: str = 'logarithmic', fan_in: int = None, no_grad: bool = True):
        super().__init__()
        assert freq_spacing_type in {'logarithmic', 'linear'}, f'Unknown frequency spacing type: {self.freq_spacing_type}'
        self.n_fourier_features = n_fourier_features
        self.fan_in = fan_in
        if fan_in is not None:
            self.out_dim = 2 * fan_in * self.n_fourier_features
        else:
            self.out_dim = None
        self.freq_spacing_type = freq_spacing_type
        self.no_grad = no_grad

    @staticmethod
    def get_output_size(input_size: int, n_fourier_features: int):
        return 2 * input_size * n_fourier_features

    def forward_fn(self, x):
        x = x.view(x.shape + (1,))
        if self.freq_spacing_type == 'logarithmic':
            feature_mul = math.pi * 2 ** torch.arange(0, self.n_fourier_features, device=x.device)
        else:
            feature_mul = math.pi * torch.arange(1, self.n_fourier_features + 1, device=x.device)
        sines = torch.sin(x * feature_mul)
        cosines = torch.cos(x * feature_mul)
        del feature_mul
        # sines.shape = cosines.shape
        out = (torch.concat([
            sines.view(sines.shape + (1,)),
            cosines.view(cosines.shape + (1,))
        ],
            dim=-1)
               .flatten(-3, -1)
               )
        del sines, cosines
        return out

    def forward(self, x) -> torch.Tensor:
        assert x.size(-1) == self.fan_in, "fan_in provided at initialization does not match size of input tensor"
        assert not (self.no_grad and x.requires_grad), "FourierLayer is set to no_grad = True, but input requires grad."
        if self.no_grad:
            with torch.no_grad():
                return self.forward_fn(x).detach()
        return self.forward_fn(x)
