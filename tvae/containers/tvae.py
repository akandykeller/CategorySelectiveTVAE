import os
import torch
import wandb
import numpy as np
from tvae.containers.grouper import Chi_Squared_from_Gaussian_2d
import torchvision

class TVAE(torch.nn.Module):
    def __init__(self, z_encoder, u_encoder, decoder, grouper):
        super(TVAE, self).__init__()
        self.z_encoder = z_encoder
        self.u_encoder = u_encoder
        self.decoder = decoder
        self.grouper = grouper

    def forward(self, x):
        z, kl_z, _, _ = self.z_encoder(x)
        u, kl_u, _, _ = self.u_encoder(x)
        s = self.grouper(z, u)
        probs_x, neg_logpx_z = self.decoder(s, x)

        return z, u, s, probs_x, kl_z, kl_u, neg_logpx_z

    def plot_decoder_weights(self, wandb_on=True):
        self.decoder.plot_weights(name='Decoder Weights', wandb_on=wandb_on)

    def plot_encoder_weights(self, wandb_on=True):
        self.z_encoder.plot_weights(name='Z-Encoder Weights', wandb_on=wandb_on)
        self.u_encoder.plot_weights(name='U-Encoder Weights', wandb_on=wandb_on)

    def get_IS_estimate(self, x, n_samples=100):
        log_likelihoods = []

        for n in range(n_samples):
            z, kl_z, log_q_z, log_p_z = self.z_encoder(x)
            u, kl_u, log_q_u, log_p_u = self.u_encoder(x)
            s = self.grouper(z, u)
            probs_x, neg_logpx_z = self.decoder(s, x)
            ll = (-1 * neg_logpx_z.flatten(start_dim=1).sum(-1, keepdim=True)
                  + log_p_z.flatten(start_dim=1).sum(-1, keepdim=True)
                  + log_p_u.flatten(start_dim=1).sum(-1, keepdim=True)
                  - log_q_z.flatten(start_dim=1).sum(-1, keepdim=True)
                  - log_q_u.flatten(start_dim=1).sum(-1, keepdim=True))
            log_likelihoods.append(ll)
        ll = torch.cat(log_likelihoods, dim=-1)
        is_estimate = torch.logsumexp(ll, -1)
        return is_estimate

    def plot_samples(self, x, e, s_dir, n_samples=100, wandb_on=True):
        assert isinstance(self.grouper, Chi_Squared_from_Gaussian_2d), "Sampling is only implemented for the 2D-TVAE"
        samples_path = os.path.join(s_dir, f'{e}_modelsamples.png')

        z = self.z_encoder.sample(x, n_samples=n_samples, cap_dim=self.grouper.spatial)
        u = self.u_encoder.sample(x, n_samples=n_samples, cap_dim=self.grouper.spatial)
        s = self.grouper(z, u)
        probs_x = self.decoder.only_decode(s)

        torchvision.utils.save_image(
            probs_x, samples_path, nrow=int(n_samples**0.5),
            padding=2, pad_value=1.0, normalize=False)

        if wandb_on:
            wandb.log({f'Samples':  wandb.Image(samples_path)})


class VAE(TVAE):
    def get_IS_estimate(self, x, n_samples=100):
        log_likelihoods = []

        for n in range(n_samples):
            z, kl_z, log_q_z, log_p_z = self.z_encoder(x)
            s = self.grouper(z, torch.zeros_like(z))
            probs_x, neg_logpx_z = self.decoder(s, x)
            ll = (-1 * neg_logpx_z.flatten(start_dim=1).sum(-1, keepdim=True)
                  + log_p_z.flatten(start_dim=1).sum(-1, keepdim=True)
                  - log_q_z.flatten(start_dim=1).sum(-1, keepdim=True))
            log_likelihoods.append(ll)
        ll = torch.cat(log_likelihoods, dim=-1)
        is_estimate = torch.logsumexp(ll, -1)
        return is_estimate

    def forward(self, x):
        z, kl_z, _, _ = self.z_encoder(x)
        u = torch.zeros_like(z)
        kl_u = torch.zeros_like(kl_z)
        s = self.grouper(z, u)
        probs_x, neg_logpx_z = self.decoder(s, x)

        return z, u, s, probs_x, kl_z, kl_u, neg_logpx_z



class TVAE_with_Preprocessor(torch.nn.Module):
    def __init__(self, preprocessor, tvae):
        super(TVAE_with_Preprocessor, self).__init__()
        self.tvae = tvae
        self.preprocessor = preprocessor

    def forward(self, x):
        preprocessed = self.preprocessor(x).detach().unsqueeze(-1).unsqueeze(-1)
        return self.tvae(preprocessed)

    def plot_decoder_weights(self, wandb_on=True):
        self.tvae.plot_decoder_weights(wandb_on=wandb_on)

    def plot_encoder_weights(self, wandb_on=True):
        self.tvae.plot_encoder_weights(wandb_on=wandb_on)

    def get_IS_estimate(self, x, n_samples=100):
        x_preprocessed = self.preprocessor(x).detach().unsqueeze(-1).unsqueeze(-1)
        return self.tvae.get_IS_estimate(x_preprocessed, n_samples)