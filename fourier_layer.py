import torch
import torch.nn as nn


class FourierLayer(nn.Module):
    def __init__(self, n_fourier_features: int, freq_spacing: str = 'logarithmic', fan_in: int = None, no_grad: bool = True):
        super().__init__()
        self.nff = n_fourier_features
        self.fan_in = fan_in
        if fan_in is not None:
            self.out_dim = 2 * fan_in * self.nff
        else:
            self.out_dim = None
        self.freq_spacing = freq_spacing
        self.no_grad = no_grad

    def forward_fn(self, x):
        x = x.view(x.shape + (1,))
        if self.freq_spacing == 'logarithmic':
            feature_mul = 2 ** torch.arange(0, self.nff, device=x.device)
        elif self.freq_spacing == 'linear':
            feature_mul = torch.arange(1, self.nff + 1, device=x.device)
        else:
            raise Exception(f'Unknown frequency spacing paradigm: {self.freq_spacing}')
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

    def forward(self, x):
        assert not (self.no_grad and x.requires_grad), "FourierLayer is set to no_grad = True, but input requires grad."
        if self.no_grad:
            with torch.no_grad():
                return self.forward_fn(x).detach()
        return self.forward_fn(x)
