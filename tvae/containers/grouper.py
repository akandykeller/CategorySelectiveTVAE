import torch
import torch.nn as nn

class Grouper(nn.Module):
    def __init__(self, model, padder):
        super(Grouper, self).__init__()
        self.model = model
        self.padder = padder

    def forward(self, z, u):
        raise NotImplementedError


class Chi_Squared_from_Gaussian_2d(Grouper):
    def __init__(self, model, padder, n_caps, cap_dim, trainable=False, mu_init=1, eps=1e-6):
        super(Chi_Squared_from_Gaussian_2d, self).__init__(model, padder)
        self.trainable = trainable
        self.correlated_mean_beta = torch.nn.Parameter(data=torch.ones(1).to('cuda')*mu_init, requires_grad=True)
        self.eps = eps
        self.n_caps = n_caps
        self.cap_dim = cap_dim
        self.spatial = int(cap_dim ** 0.5)

        nn.init.ones_(self.model.weight)

        if not trainable:
            self.model.weight.requires_grad = False

    def forward(self, z, u):      
        u_spatial = u.view(u.shape[0], 1, self.spatial, self.spatial, -1) ** 2.0
        u_spatial_padded = self.padder(u_spatial)
        v = self.model(u_spatial_padded).squeeze(1)

        std = 1.0 / torch.sqrt(v + self.eps)

        s =  (z + self.correlated_mean_beta) * std.view(z.shape)
        return s


class NonTopographic_Capsules1d(Grouper):
    def __init__(self, model, padder, n_caps, cap_dim, n_transforms,
                 trainable=False, eps=1e-6):
        super(NonTopographic_Capsules1d, self).__init__(model, padder)
        self.n_caps = n_caps
        self.cap_dim = cap_dim
        self.n_t = n_transforms
        self.trainable = trainable
        self.eps = eps
       
    def forward(self, z, u):
        s = z
        return s
