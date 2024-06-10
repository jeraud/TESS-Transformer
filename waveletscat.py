import torch
import torch.nn as nn
from kymatio.torch import Scattering1D

class WaveletScatteringTransform(nn.Module):
    def __init__(self, device, T=16384, J=8, Q=12, log_eps=1e-6, **kwargs):
        """
        :param
        """
        super().__init__()

        self.log_eps = log_eps
        self.T = T
        self.J = J
        self.Q = Q
        self.device = device

        self.scattering = Scattering1D(J,T,Q).to(self.device)

        #def forward(self, x):
        #    x = self.scattering(x) 
        #return x


    def transform(self, x):
        """
        :param x: Input sequence (B, T, 1).
        :return: Output sequence (B, T, n_out).
        """

        Sx_all = self.scattering.forward(x)

        #Since it does not carry useful information, we remove the zeroth-order scattering coefficients,
        #which are always placed in the first channel of the scattering transform.
        Sx_all = Sx_all[:,1:,:]

        # To increase discriminability, we take the logarithm of the scattering
        # coefficients (after adding a small constant to make sure nothing blows up
        # when scattering coefficients are close to zero). This is known as the
        # log-scattering transform.
        Sx_all = torch.log(torch.abs(Sx_all) + self.log_eps)

        #We then average along the last dimension (time) to get a time-shift invariant representation.
        Sx_all = torch.mean(Sx_all, dim=-1)

        return Sx_all
    