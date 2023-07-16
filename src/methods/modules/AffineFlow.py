import sys
sys.path.append('')
from torch import nn
import torch
from torch.distributions.normal import Normal
from src.methods.modules import MaskedMLP

class AffineFlow(nn.Module):
    def __init__(self, d_in, d_out, dh):
        super().__init__()
        self.d_out = d_out
        self.eta1 = MaskedMLP(d_in=d_in * d_out, d_out=d_out, n_groups=d_out, hidden_sizes=dh * d_out, mask_type='grouped')
        self.eta2 = MaskedMLP(d_in=d_in * d_out, d_out=d_out, n_groups=d_out, hidden_sizes=dh * d_out, mask_type='grouped')
        self.dist = Normal(0., 1.)

    def forward(self, X, Z):
        Z = Z.repeat(1, self.d_out)

        e1 = self.eta1(Z)
        e2 = -torch.clip(self.eta2(Z), -4, 4).exp()
        t = -e1 / (2 * e2)
        s = (-1 / (2 * e2)).log() / 2 # (ln0.5 - self.eta2) / 2

        e = (X - t) / s.exp()
        ll = e1 * X + e2 * X * X + e1 * e1 / (4 * e2) + torch.log(torch.sqrt(-2 * e2))

        return e, ll, s, t