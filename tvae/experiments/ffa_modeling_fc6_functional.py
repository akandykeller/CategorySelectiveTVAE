import os
import torch
from torch import optim
from torch.optim.lr_scheduler import StepLR
from torch import nn
from torch.nn import functional as F

from tvae.data.imagenet import Multiset_Preprocessor
from tvae.containers.tvae import TVAE, TVAE_with_Preprocessor
from tvae.models.mlp import FC_Encoder, FC_Decoder
from tvae.models.alexnet import create_alexnet_fc6
from tvae.containers.encoder import Gaussian_Encoder
from tvae.containers.decoder import Gaussian_Decoder
from tvae.containers.grouper import Chi_Squared_from_Gaussian_2d
from tvae.utils.logging import configure_logging, get_dirs
from tvae.utils.train_loops import train_epoch_imagenet, eval_epoch_many_imagenet

def create_model(s_dim, group_kernel, mu_init):
    z_encoder = Gaussian_Encoder(FC_Encoder(n_in=9216, n_out=s_dim),
                                 loc=0.0, scale=1.0)

    u_encoder = Gaussian_Encoder(FC_Encoder(n_in=9216, n_out=s_dim),                                
                                 loc=0.0, scale=1.0)

    decoder = Gaussian_Decoder(FC_Decoder(n_in=s_dim, n_out=9216))

    grouper = Chi_Squared_from_Gaussian_2d(nn.ConvTranspose3d(in_channels=1, out_channels=1,
                                          kernel_size=group_kernel, 
                                          padding=(2*(group_kernel[0] // 2), 
                                                   2*(group_kernel[1] // 2),
                                                   2*(group_kernel[2] // 2)),
                                          stride=(1,1,1), padding_mode='zeros', bias=False),
                      lambda x: F.pad(x, (group_kernel[2] // 2, group_kernel[2] // 2,
                                          group_kernel[1] // 2, group_kernel[1] // 2,
                                          group_kernel[0] // 2, group_kernel[0] // 2), 
                                          mode='circular'),
                       n_caps=1, cap_dim=s_dim,
                       mu_init=mu_init)
    
    tvae = TVAE(z_encoder, u_encoder, decoder, grouper)
    preprocessing_model = create_alexnet_fc6()

    return TVAE_with_Preprocessor(preprocessing_model, tvae)


def main():
    config = {
        'wandb_on': True,
        'lr': 1e-5,
        'momentum': 0.9,
        'batch_size': 128,
        'max_epochs': 100,
        'eval_epochs': 1,
        'train_datadir': 'TRAIN_DATA_DIR',
        'test_datadirs': [('Animate', 'ANIMATE_DATA_DIR'),
                          ('Inanimate', 'INANIMATE_DATA_DIR')],
        # 'test_datadirs': [('Big', 'BIG_DATA_DIR'),
                        #   ('Small', 'SMALL_DATA_DIR')],
        'seed': 1,
        'k': 25,
        's_dim': 64*64,
        'mu_init': 40.0,
        'n_is_samples': 10,
        }

    name = 'TVAE_Animacy_vs_Inanimacy'
    config['savedir'], _, config['wandb_dir'] = get_dirs()

    savepath = os.path.join(config['savedir'], name)
    preprocessor = Multiset_Preprocessor(config)
    train_loader, test_loaders = preprocessor.get_dataloaders()


    model = create_model(s_dim=config['s_dim'], group_kernel=(config['k'],config['k'],1), mu_init=config['mu_init'])
    model.to('cuda')
    
    log, checkpoint_path = configure_logging(config, name, model)
    # load_checkpoint_path = 'checkpoint.tar'
    # model.load_state_dict(torch.load(load_checkpoint_path))

    optimizer = optim.SGD(model.tvae.parameters(), 
                           lr=config['lr'],
                           momentum=config['momentum'])
    scheduler = StepLR(optimizer, step_size=1, gamma=1.0)

    for e in range(config['max_epochs']):
        log('Epoch', e)

        total_loss, total_neg_logpx_z, total_kl, num_batches = train_epoch_imagenet(
                                                                     model,
                                                                     optimizer, 
                                                                     train_loader, log,
                                                                     savepath, e, eval_batches=1000,
                                                                     plot_weights=False,
                                                                     wandb_on=config['wandb_on'])

        log("Epoch Avg Loss", total_loss / num_batches)
        log("Epoch Avg -LogP(x|z)", total_neg_logpx_z / num_batches)
        log("Epoch Avg KL", total_kl / num_batches)
        scheduler.step()

        torch.save(model.state_dict(), checkpoint_path)

        if e % config['eval_epochs'] == 0:
            total_loss, total_neg_logpx_z, total_kl, total_is_estimate, num_batches = eval_epoch_many_imagenet(
                                                                                        model,
                                                                                        test_loaders,
                                                                                        ['#00b4cc', '#d1621d'],
                                                                                        ['Animate', 'Inaimate'],
                                                                                        log, savepath, e, 
                                                                                        n_is_samples=config['n_is_samples'],
                                                                                        plot_maxact=False, 
                                                                                        plot_class_selectivity=True,
                                                                                        wandb_on=config['wandb_on'])
            log("Val Avg Loss", total_loss / num_batches)
            log("Val Avg -LogP(x|z)", total_neg_logpx_z / num_batches)
            log("Val Avg KL", total_kl / num_batches)
            log("Val IS Estiamte", total_is_estimate / num_batches)

if __name__ == '__main__':
    main()