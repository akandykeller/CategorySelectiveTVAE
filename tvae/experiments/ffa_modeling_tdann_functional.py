import os
import torch
import numpy as np
from torch import optim
from torch.optim.lr_scheduler import StepLR
from torch import nn
from torch.nn import functional as F

from tvae.data.imagenet import Multiset_Preprocessor
from tvae.models.alexnet import create_alexnet_classifier
from tvae.utils.logging import configure_logging, get_dirs
from tvae.utils.train_loops import train_classification_epoch_imagenet, eval_classification_epoch_many_imagenet

def main():
    config = {
        'wandb_on': True,
        'lr': 1e-3,
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
        's_dim': 64*64,
        'mu_init': 0.0,
        'n_is_samples': 10,
        'fc6_scl_weight': 10*0.000000059604645,
        'fc7_scl_weight': 10*0.000000059604645,
        'weight_decay': 0.0005,
        'D_path':  'datasets/D.npy',
        }

    name = 'TDANN_Animacy_vs_Inanimacy'
    config['savedir'], _, config['wandb_dir'] = get_dirs()

    savepath = os.path.join(config['savedir'], name)
    preprocessor = Multiset_Preprocessor(config)
    train_loader, test_loaders = preprocessor.get_dataloaders()


    model = create_alexnet_classifier()
    
    D = np.load(config['D_path'])
    D = torch.from_numpy(D).to('cuda')

    log, checkpoint_path = configure_logging(config, name, model)
    # load_checkpoint_path = 'checkpoint.tar'
    # model.load_state_dict(torch.load(load_checkpoint_path))
    model.to('cuda')

    optimizer = optim.SGD(model.parameters(), 
                           lr=config['lr'],
                           momentum=config['momentum'],
                           weight_decay=config['weight_decay'])
    scheduler = StepLR(optimizer, step_size=100, gamma=0.1, last_epoch=-1)
    criterion = nn.CrossEntropyLoss(reduction='none')

    for e in range(config['max_epochs']):
        log('Epoch', e)

        total_class_loss, total_fc6_scl, total_fc7_scl, num_batches = train_classification_epoch_imagenet(
                                                                     model,
                                                                     optimizer, 
                                                                     criterion,
                                                                     train_loader, log,
                                                                     savepath, e, 
                                                                     fc6_scl_weight=config['fc6_scl_weight'],
                                                                     fc7_scl_weight=config['fc7_scl_weight'],
                                                                     D=D,
                                                                     eval_batches=1000,
                                                                     wandb_on=config['wandb_on'])

        log("Epoch Avg Class Loss", total_class_loss / num_batches)
        log("Epoch Avg FC6", total_fc6_scl / num_batches)
        log("Epoch Avg FC7", total_fc7_scl / num_batches)
        scheduler.step()

        torch.save(model.state_dict(), checkpoint_path)

        if e % config['eval_epochs'] == 0:
            total_class_loss, total_fc6_scl, total_fc7_scl, acc, num_batches = eval_classification_epoch_many_imagenet(
                                                                                        model,
                                                                                        criterion,
                                                                                        test_loaders,
                                                                                        ['#00b4cc', '#d1621d'], 
                                                                                        ['Animate', 'Inanimate'],
                                                                                        log, savepath, e, 
                                                                                        D=D,
                                                                                        n_is_samples=config['n_is_samples'],
                                                                                        plot_maxact=False, 
                                                                                        plot_class_selectivity=True,
                                                                                        wandb_on=config['wandb_on'])
            log("Val Avg Class Loss", total_class_loss / num_batches)
            log("Val Avg FC6", total_fc6_scl / num_batches)
            log("Val Avg FC7", total_fc7_scl / num_batches)
            log("Val Accuracy", acc)

if __name__ == '__main__':
    main()