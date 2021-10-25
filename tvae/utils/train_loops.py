import os
import torch
from tvae.utils.vis import Plot_ClassActMap_SNS_Continuous, plot_recon, Plot_MaxActImg, Plot_ClassActMap, Plot_ClassActMap_SNS, Plot_ClassActMap_SNS_MultiColor, Plot_Single_Img_Map
from tvae.utils.correlations import Plot_Covariance_Matrix
from tvae.utils.losses import Spatial_loss
import numpy as np
from tqdm import tqdm
import wandb

def train_epoch_imagenet(model, optimizer, train_loader, log, 
                savepath, epoch, eval_batches=300, plot_weights=False, 
                plot_reconstruction=False, anneal_kl=False, wandb_on=True,
                train_batches_per_epoch=None):
    total_loss = 0
    total_kl = 0
    total_neg_logpx_z = 0
    num_batches = 0
    b_idx = epoch * len(train_loader)

    model.train()
    for x, label in tqdm(train_loader):
        optimizer.zero_grad()
        x = x.float().to('cuda')

        x_batched = x.view(-1, *x.shape[-3:])  # batch transforms
        z, u, s, probs_x, kl_z, kl_u, neg_logpx_z = model(x_batched)

        if anneal_kl:
            kl_weight = min((np.tanh(b_idx / 200.0 - 4.0) + 1)/2.0, 1.0) 
        else:
            kl_weight = 1.0

        avg_KLD = (kl_z.sum() + kl_u.sum()) / x_batched.shape[0]
        avg_neg_logpx_z = neg_logpx_z.sum() / x_batched.shape[0]
        loss = avg_neg_logpx_z + avg_KLD * kl_weight

        loss.backward()
        optimizer.step()    

        total_loss += loss
        total_neg_logpx_z += avg_neg_logpx_z
        total_kl += avg_KLD
        num_batches += 1
        b_idx = epoch * len(train_loader) + num_batches

        if b_idx % eval_batches == 0:
            log('Train Total Loss', loss)
            log('Train -LogP(x|z)', avg_neg_logpx_z)
            log('Train KLD', avg_KLD)

            if plot_weights:
                model.plot_decoder_weights(wandb_on=wandb_on)
                model.plot_encoder_weights(wandb_on=wandb_on)

            if plot_reconstruction:
                plot_recon(x_batched, 
                           probs_x.view(x_batched.shape), 
                           os.path.join(savepath, 'samples'),
                           b_idx, wandb_on=wandb_on)

    return total_loss, total_neg_logpx_z, total_kl, num_batches



def eval_epoch_imagenet(model, 
               test_loader_obj, test_loader_other, 
               log, savepath, epoch, n_is_samples=100, 
               plot_maxact=False, plot_class_selectivity=False, 
               plot_cov=False, 
               max_val_batches=9999999999,
               wandb_on=True):
    total_loss = 0.0
    total_kl = 0
    total_neg_logpx_z = 0
    total_is_estimate = 0.0
    all_s = []
    all_labels = []

    model.eval()
    with torch.no_grad():
        for loader_idx, loader in enumerate([test_loader_obj, test_loader_other]):
            num_batches = 0
            for x, label in tqdm(loader):
                x = x.float().to('cuda')

                x_batched = x.view(-1, *x.shape[-3:])  # batch transforms
                z, u, s, probs_x, kl_z, kl_u, neg_logpx_z = model(x_batched)
                avg_KLD = (kl_z.sum() + kl_u.sum()) / x_batched.shape[0]
                avg_neg_logpx_z = neg_logpx_z.sum() / x_batched.shape[0]

                all_s.append(s.cpu().detach())
                all_labels.append(torch.ones_like(label).cpu().detach() * loader_idx)

                loss = avg_neg_logpx_z + avg_KLD
                total_loss += loss
                total_neg_logpx_z += avg_neg_logpx_z
                total_kl += avg_KLD

                # is_estimate = model.get_IS_estimate(x_batched, n_samples=n_is_samples)
                is_estimate = torch.tensor(0.0)
                total_is_estimate += is_estimate.sum() / x_batched.shape[0]

                num_batches += 1

    if plot_cov or plot_maxact or plot_class_selectivity:
        all_s = torch.cat(all_s, 0)
        all_labels = torch.cat(all_labels, 0)

    if plot_class_selectivity:
        Plot_ClassActMap_SNS(all_s, all_labels, os.path.join(savepath, 'samples'), epoch, wandb_on=wandb_on)
        Plot_ClassActMap_SNS_Continuous(all_s, all_labels, os.path.join(savepath, 'samples'), epoch, wandb_on=wandb_on)


    return total_loss, total_neg_logpx_z, total_kl, total_is_estimate, num_batches


def eval_epoch_many_imagenet(model, 
               test_loaders,
               colors,
               class_names,
               log, savepath, epoch, n_is_samples=100, 
               plot_maxact=False, plot_class_selectivity=False, 
               plot_cov=False, 
               wandb_on=True,
               background_color='#000000', 
               mix_color='#4d4d4d',
               select_thresh=0.85):
    total_loss = 0.0
    total_kl = 0
    total_neg_logpx_z = 0
    total_is_estimate = 0.0
    all_x = []
    all_s = []
    all_labels = []

    single_img_maps = []

    model.eval()
    with torch.no_grad():
        for loader_idx, loader in enumerate(test_loaders):
            num_batches = 0
            for x, label in tqdm(loader):
                x = x.float().to('cuda')

                x_batched = x.view(-1, *x.shape[-3:])  # batch transforms
                z, u, s, probs_x, kl_z, kl_u, neg_logpx_z = model(x_batched)
                avg_KLD = (kl_z.sum() + kl_u.sum()) / x_batched.shape[0]
                avg_neg_logpx_z = neg_logpx_z.sum() / x_batched.shape[0]

                all_s.append(s.cpu().detach())
                all_labels.append(torch.ones_like(label).cpu().detach() * loader_idx)

                loss = avg_neg_logpx_z + avg_KLD
                total_loss += loss
                total_neg_logpx_z += avg_neg_logpx_z
                total_kl += avg_KLD

                is_estimate = model.get_IS_estimate(x_batched, n_samples=n_is_samples)
                # is_estimate = torch.tensor(0.0)
                total_is_estimate += is_estimate.sum() / x_batched.shape[0]

                num_batches += 1

                if len(single_img_maps) <= loader_idx:
                    img_idx = np.random.randint(x.shape[0])
                    single_img_maps.append((x[img_idx], s[img_idx], class_names[loader_idx]))

    if plot_cov or plot_maxact or plot_class_selectivity:
        all_s = torch.cat(all_s, 0)
        all_labels = torch.cat(all_labels, 0)

    if plot_class_selectivity:
        Plot_ClassActMap_SNS_Continuous(all_s[(all_labels.view(-1) == 1) |  (all_labels.view(-1) == 0)], 
                                        all_labels[(all_labels.view(-1) == 1) |  (all_labels.view(-1) == 0)],
                                        os.path.join(savepath, 'samples'), 
                                        epoch, wandb_on=wandb_on, name=f'Class_Selec_{class_names[1]}_v_{class_names[0]}')
        selectivity_maps = Plot_ClassActMap_SNS_MultiColor(all_s, all_labels,  os.path.join(savepath, 'samples'), epoch, 
                                        colors=colors, class_names=class_names, 
                                        thresh=select_thresh, wandb_on=True, name='Class_Selectivity',
                                        background_color=background_color, mix_color=mix_color)
        Plot_Single_Img_Map(single_img_maps, os.path.join(savepath, 'samples'), 
                            epoch, wandb_on=wandb_on, name='Single_Img_Map')

    return total_loss, total_neg_logpx_z, total_kl, total_is_estimate, num_batches




def train_classification_epoch_imagenet(model, optimizer, criterion, train_loader, log, 
                savepath, epoch, fc6_scl_weight, fc7_scl_weight, D, eval_batches=300, 
                plot_weights=False, plot_reconstruction=False, anneal_kl=False, wandb_on=True):
    total_class_loss = 0
    total_fc6_scl = 0
    total_fc7_scl = 0
    num_ex = 0
    num_correct = 0
    num_batches = 0
    b_idx = epoch * len(train_loader)

    model.train()
    for x, label in tqdm(train_loader):
        optimizer.zero_grad()
        x = x.float().to('cuda')

        x_batched = x.view(-1, *x.shape[-3:])  # batch transforms
        z, fc6, fc7 = model(x_batched)

        class_loss = criterion(z, label.to('cuda'))
        fc6_scl = Spatial_loss(fc6, D)
        fc7_scl = Spatial_loss(fc7, D)

        avg_class_loss = (class_loss.sum()) / x_batched.shape[0]
        avg_fc6_scl = fc6_scl
        avg_fc7_scl = fc7_scl

        loss = avg_class_loss + fc6_scl_weight * avg_fc6_scl + fc7_scl_weight * avg_fc7_scl

        loss.backward()
        optimizer.step()    

        _, predicted = torch.max(z, 1)
        num_ex += label.size(0)
        num_correct += (predicted.cpu().detach() == label).sum().item()

        total_class_loss += avg_class_loss
        total_fc6_scl += avg_fc6_scl
        total_fc7_scl += avg_fc7_scl
        num_batches += 1
        b_idx = epoch * len(train_loader) + num_batches

        if b_idx % eval_batches == 0:
            log('Train Total Loss', loss)
            log('Train Class Loss', avg_class_loss)
            log('Train FC6 Loss', avg_fc6_scl)
            log('Train FC7 Loss', avg_fc7_scl)
            log('Train Acc', num_correct / float(num_ex))

        num_batches += 1

    return total_class_loss, total_fc6_scl, total_fc7_scl, num_batches



def eval_classification_epoch_many_imagenet(model, criterion,
               test_loaders,
               colors,
               class_names,               
               log, savepath, epoch, D, n_is_samples=100, 
               plot_maxact=False, plot_class_selectivity=False, 
               plot_cov=False, 
               wandb_on=True):
    total_class_loss = 0
    total_fc6_scl = 0
    total_fc7_scl = 0
    num_ex = 0
    num_correct = 0

    all_fc6 = []
    all_fc7 = []
    all_labels = []

    model.eval()
    with torch.no_grad():
        for loader_idx, loader in enumerate(test_loaders):
            num_batches = 0
            for x, label in tqdm(loader):
                x = x.float().to('cuda')

                x_batched = x.view(-1, *x.shape[-3:])  # batch transforms
                z, fc6, fc7 = model(x_batched)

                class_loss = criterion(z, label.to('cuda'))
                fc6_scl = Spatial_loss(fc6, D)
                fc7_scl = Spatial_loss(fc7, D)

                avg_class_loss = (class_loss.sum()) / x_batched.shape[0]
                avg_fc6_scl = fc6_scl
                avg_fc7_scl = fc7_scl
                
                all_fc6.append(fc6.cpu().detach())
                all_fc7.append(fc7.cpu().detach())
                all_labels.append(torch.ones_like(label).cpu().detach() * loader_idx)

                _, predicted = torch.max(z, 1)
                num_ex += label.size(0)
                num_correct += (predicted.cpu().detach() == label).sum().item()

                total_class_loss += avg_class_loss
                total_fc6_scl += avg_fc6_scl
                total_fc7_scl += avg_fc7_scl

                num_batches += 1

    if plot_cov or plot_maxact or plot_class_selectivity:
        all_fc6 = torch.cat(all_fc6, 0)
        all_fc7 = torch.cat(all_fc7, 0)
        all_labels = torch.cat(all_labels, 0)

    if plot_class_selectivity:
        Plot_ClassActMap_SNS_Continuous(all_fc7[(all_labels.view(-1) == 1) |  (all_labels.view(-1) == 0)], 
                                        all_labels[(all_labels.view(-1) == 1) |  (all_labels.view(-1) == 0)],
                                        os.path.join(savepath, 'samples'), 
                                        epoch, wandb_on=wandb_on, name=f'Class_Selec_FC7_{class_names[1]}_v_{class_names[0]}')
        selectivity_maps = Plot_ClassActMap_SNS_MultiColor(all_fc7, all_labels,  os.path.join(savepath, 'samples'), epoch, 
                                        colors=colors, class_names=class_names, 
                                        thresh=0.85, wandb_on=True, name='Class_Selectivity_FC7')


        Plot_ClassActMap_SNS_Continuous(all_fc6[(all_labels.view(-1) == 1) |  (all_labels.view(-1) == 0)], 
                                        all_labels[(all_labels.view(-1) == 1) |  (all_labels.view(-1) == 0)],
                                        os.path.join(savepath, 'samples'), 
                                        epoch, wandb_on=wandb_on, name=f'Class_Selec_FC6_{class_names[1]}_v_{class_names[0]}')
        selectivity_maps = Plot_ClassActMap_SNS_MultiColor(all_fc6, all_labels,  os.path.join(savepath, 'samples'), epoch, 
                                        colors=colors, class_names=class_names, 
                                        thresh=0.85, wandb_on=True, name='Class_Selectivity_FC6')

    return total_class_loss, total_fc6_scl, total_fc7_scl, float(num_correct)/float(num_ex), num_batches