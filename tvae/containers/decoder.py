import torch
import torch.nn as nn
from torch.distributions.bernoulli import Bernoulli
from torch.distributions.normal import Normal

class Decoder(nn.Module):
    def __init__(self, model):
        super(Decoder, self).__init__()
        self.model = model

    def forward(self, x):
        raise NotImplementedError


class Bernoulli_Decoder(Decoder):
    def __init__(self, model):
        super(Bernoulli_Decoder, self).__init__(model)

    def forward(self, z, x):
        probs_x = torch.clamp(self.model(z), 0, 1)
        p = Bernoulli(probs=probs_x)
        neg_logpx_z = -1 * p.log_prob(x)

        return probs_x, neg_logpx_z

    def only_decode(self, z):
        probs_x = torch.clamp(self.model(z), 0, 1)
        return probs_x


class Gaussian_Decoder(Decoder):
    def __init__(self, model, scale=1.0):
        super(Gaussian_Decoder, self).__init__(model)
        self.scale = torch.tensor([scale]).to('cuda')

    def forward(self, z, x):
        mu_x = self.model(z)
        p = Normal(loc=mu_x, scale=self.scale)
        neg_logpx_z = -1 * p.log_prob(x)

        return mu_x, neg_logpx_z

    def only_decode(self, z):
        mu_x = self.model(z)
        return mu_x


class Dummy_Decoder(Decoder):
    def __init__(self, model, scale=1.0):
        super(Dummy_Decoder, self).__init__(model)
        self.scale = torch.tensor([scale]).to('cuda')

    def forward(self, z, x):
        return z, torch.tensor(0.0)

    def only_decode(self, z):
        return z