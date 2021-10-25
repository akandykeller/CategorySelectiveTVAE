import torchvision
import os
import wandb 
import torch
import matplotlib.pyplot as plt
import matplotlib.colors as colors
import numpy as np
import seaborn as sns
from pylab import *
from matplotlib.colors import ListedColormap

def plot_recon(x, xhat, s_dir, e, wandb_on, max_plot=100):
    x_path = os.path.join(s_dir, f'{e}_x.png')
    xhat_path = os.path.join(s_dir, f'{e}_xrecon.png')
    diff_path = os.path.join(s_dir, f'{e}_recon_diff.png')

    xhat = xhat[:max_plot]
    x = x[:max_plot]

    n_row = int(x.shape[0] ** 0.5)

    os.makedirs(s_dir, exist_ok=True)
    torchvision.utils.save_image(
        xhat, xhat_path, nrow=n_row,
        padding=2, normalize=False)

    torchvision.utils.save_image(
        x, x_path, nrow=n_row,
        padding=2, normalize=False)

    xdiff = torch.abs(x - xhat)

    torchvision.utils.save_image(
        xdiff, diff_path, nrow=n_row,
        padding=2, normalize=False)

    if wandb_on:
        wandb.log({'X Original':  wandb.Image(x_path)})
        wandb.log({'X Recon':  wandb.Image(xhat_path)})
        wandb.log({'Recon diff':  wandb.Image(diff_path)})

def Plot_MaxActImg(all_s, all_x, s_dir, e, wandb_on):
    max_xs = []
    for s_idx in range(all_s.shape[1]):
        max_idx = torch.max(torch.abs(all_s[:, s_idx]), 0)[1]
        max_xs.append(all_x[max_idx].squeeze().unsqueeze(0).unsqueeze(0))
    
    path = os.path.join(s_dir, f'{e}_maxactimg.png')
    os.makedirs(s_dir, exist_ok=True)

    x_image = torch.cat(max_xs)

    sq = int(float(all_s.shape[1]) ** 0.5)

    torchvision.utils.save_image(
        x_image, path, nrow=sq,
        padding=2, normalize=False)

    if wandb_on:
        wandb.log({'Max_Act_Img':  wandb.Image(path)})


def Plot_ClassActMap(all_s, all_labels, s_dir, e, n_class=10, thresh=0.85, wandb_on=True):
    s_dim = all_s[0].shape[0]
    sq = int(float(s_dim)**0.5)
    class_selectivity = torch.ones_like(all_s[0]) * -1

    for c in range(n_class):
        s_c = all_s[all_labels.view(-1) == c]
        s_other = all_s[all_labels.view(-1) != c]

        s_mean_c = s_c.mean(0).squeeze()
        s_mean_other = s_other.mean(0).squeeze()
        s_var_c = s_c.var(0).squeeze()
        s_var_other = s_other.var(0).squeeze()

        dprime = (s_mean_c - s_mean_other) / torch.sqrt((s_var_c + s_var_other)/2.0)

        class_selectivity[dprime >= thresh] = c

    class_selectivity = class_selectivity.view(sq, sq)
    fig, ax = plt.subplots()
    ax.matshow(class_selectivity)
    for i in range(sq):
        for j in range(sq):
            c = class_selectivity[j,i].item()
            ax.text(i, j, str(int(c)), va='center', ha='center')

    if wandb_on:
        wandb.log({'Class Selectivity': wandb.Image(plt)})
    else:
        path = os.path.join(s_dir, f'{e}_classactmap.png')
        plt.savefig(path)
    plt.close('all')

def Plot_ClassActMap_SNS(all_s, all_labels, s_dir, e, n_class=2, thresh=0.85, wandb_on=True, name='Class_Selectivity'):
    s_dim = all_s[0].shape[0]
    sq = int(float(s_dim)**0.5)
    class_selectivity = torch.zeros_like(all_s[0])
    class_to_idx = [-1, 1]

    for c in range(n_class):
        s_c = all_s[all_labels.view(-1) == c]
        s_other = all_s[all_labels.view(-1) != c]

        s_mean_c = s_c.mean(0).squeeze()
        s_mean_other = s_other.mean(0).squeeze()
        s_var_c = s_c.var(0).squeeze()
        s_var_other = s_other.var(0).squeeze()

        dprime = (s_mean_c - s_mean_other) / torch.sqrt((s_var_c + s_var_other)/2.0)

        class_selectivity[dprime >= thresh] = class_to_idx[c]

    class_selectivity = class_selectivity.view(sq, sq)

    # cMap = ListedColormap(['#00cc66','#7575a3','#ffcc00'])
    cMap = ListedColormap(['#00b4cc', '#4d4d4d', '#d1621d'])

    plt.figure(figsize=(8,6))
    ax = sns.heatmap(data=class_selectivity,
                     linewidths=0.1,
                     vmin=-1.0,
                     vmax=1.0,
                     center=0.0,
                    #  linecolor="#666699",
                     linecolor="#363636",
                     cmap=cMap,
                     cbar_kws={"ticks":[-0.7, 0., 0.7]})

    cbar = ax.collections[0].colorbar
    cbar.set_ticks([-0.7, 0., 0.7])
    cbar.set_ticklabels( ['Objects', 'None', 'Faces'])
    cbar.ax.tick_params(labelsize=15)
    plt.xticks([])
    plt.yticks([])
    plt.show()

    if wandb_on:
        wandb.log({name: wandb.Image(plt)})
    else:
        path = os.path.join(s_dir, f'{e}_{name}.png')
        plt.savefig(path)
    plt.close('all')


def Plot_ClassActMap_SNS_Continuous(all_s, all_labels, s_dir, e, n_class=2, wandb_on=True, name='Class_Selectivity_Continuous'):
    s_dim = all_s[0].shape[0]
    sq = int(float(s_dim)**0.5)

    s_c = all_s[all_labels.view(-1) == 1]
    s_other = all_s[all_labels.view(-1) == 0]

    s_mean_c = s_c.mean(0).squeeze()
    s_mean_other = s_other.mean(0).squeeze()
    s_var_c = s_c.var(0).squeeze()
    s_var_other = s_other.var(0).squeeze()

    dprime = (s_mean_c - s_mean_other) / torch.sqrt((s_var_c + s_var_other)/2.0)

    dprime = dprime.view(sq,sq)

    # cMap = ListedColormap(['#00cc66','#7575a3','#ffcc00'])
    # cMap = ListedColormap(['#d1621d', '#4d4d4d', '#00b4cc'])

    plt.figure(figsize=(8,6))
    ax = sns.heatmap(data=dprime,
                     linewidths=0.1,
                     vmin=-1.0,
                     vmax=1.0,
                     center=0.0,
                    #  linecolor="#666699",
                     linecolor="#363636")
                    #  cmap=cMap,
                    #  cbar_kws={"ticks":[-0.7, 0., 0.7]})

    cbar = ax.collections[0].colorbar
    cbar.set_ticks([-0.7, 0., 0.7])
    cbar.set_ticklabels( ['Objects', 'None', 'Faces'])
    cbar.ax.tick_params(labelsize=15)
    plt.xticks([])
    plt.yticks([])
    plt.show()

    if wandb_on:
        wandb.log({name: wandb.Image(plt)})
    else:
        path = os.path.join(s_dir, f'{e}_{name}.png')
        plt.savefig(path)
    plt.close('all')


def Plot_ClassActMap_SNS_MultiColor(all_s, all_labels, s_dir, e, colors, class_names, thresh=0.85, wandb_on=True, name='Class_Selectivity',
                                    background_color='#000000', mix_color='#4d4d4d'):
    s_dim = all_s[0].shape[0]
    sq = int(float(s_dim)**0.5)
    n_classes = max(all_labels) + 1

    assert n_classes == len(colors) == len(class_names)
    selectivity_maps = []
    dprime_maps = []
    joint_map = torch.zeros_like(all_s[0])

    # Compute Selectivity maps
    for c in range(n_classes):
        class_selectivity = torch.zeros_like(all_s[0])

        s_c = all_s[all_labels.view(-1) == c]
        s_other = all_s[all_labels.view(-1) != c]

        s_mean_c = s_c.mean(0).squeeze()
        s_mean_other = s_other.mean(0).squeeze()
        s_var_c = s_c.var(0).squeeze()
        s_var_other = s_other.var(0).squeeze()

        dprime = (s_mean_c - s_mean_other) / torch.sqrt((s_var_c + s_var_other)/2.0)

        class_selectivity[dprime >= thresh] = 1

        for idx, d in enumerate(dprime):
            if joint_map[idx] == 0 and d >= thresh:
                joint_map[idx] = c + 2
            elif joint_map[idx] != 0 and d >= thresh:
                joint_map[idx] = 1

        selectivity_maps.append(class_selectivity)
        dprime_maps.append(dprime)


    # Make individual plots
    for sm, dm, color, class_name in zip(selectivity_maps, dprime_maps, colors, class_names):
        # Make Discrete Plots
        sm = sm.view(sq, sq)
        cMap = ListedColormap([background_color, color])
        plt.figure(figsize=(8,6))
        ax = sns.heatmap(data=sm,
                        linewidths=0.1,
                        vmin=0.0,
                        vmax=1.0,
                        center=0.5,
                        #  linecolor="#666699",
                        linecolor="#363636",
                        cmap=cMap,
                        cbar_kws={"ticks":[0.25, 0.75]})
        cbar = ax.collections[0].colorbar
        cbar.set_ticks([0.25, 0.75])
        cbar.set_ticklabels( ['None', class_name])
        cbar.ax.tick_params(labelsize=15)
        plt.xticks([])
        plt.yticks([])
        plt.show()
        if wandb_on:
            wandb.log({f'{name}_{class_name}': wandb.Image(plt)})
        else:
            path = os.path.join(s_dir, f'{name}_{class_name}_{e}.png')
            plt.savefig(path)
        plt.close('all')

        # Make Continuous Plots
        dm = dm.view(sq, sq)
        plt.figure(figsize=(8,6))
        ax = sns.heatmap(data=dm,
                        linewidths=0.1,
                        vmin=-1.0,
                        vmax=1.0,
                        center=0.0,
                        linecolor="#363636")
        cbar = ax.collections[0].colorbar
        cbar.set_ticks([-0.7, 0., 0.7])
        cbar.set_ticklabels( ['Control', 'None', class_name])
        cbar.ax.tick_params(labelsize=15)
        plt.xticks([])
        plt.yticks([])
        plt.show()

        if wandb_on:
            wandb.log({f'{name}-Continuous_{class_name}': wandb.Image(plt)})
        else:
            path = os.path.join(s_dir,'{name}-Continuous_{class_name}.png')
            plt.savefig(path)
        plt.close('all')


    # Make Joint plot
    joint_map = joint_map.view(sq, sq)
    cMap = ListedColormap([background_color, mix_color] +  colors)
    plt.figure(figsize=(8,6))
    ax = sns.heatmap(data=joint_map,
                    linewidths=0.1,
                    vmin=0.0,
                    vmax=float(n_classes+1),
                    center=float(n_classes+1) / 2.0,
                    #  linecolor="#666699",
                    linecolor="#363636",
                    cmap=cMap,
                    cbar_kws={"ticks":[x + 0.5 for x in range(0, n_classes+1)]})

    cbar = ax.collections[0].colorbar
    cbar.set_ticks([x + 0.5 for x in range(0, n_classes+1)])
    cbar.set_ticklabels( ['None', 'Mixed'] + class_names)
    cbar.ax.tick_params(labelsize=15)
    plt.xticks([])
    plt.yticks([])
    plt.show()

    if wandb_on:
        wandb.log({f'{name}_Joint': wandb.Image(plt)})
    else:
        path = os.path.join(s_dir, f'{name}_Joint_{e}.png')
        plt.savefig(path)
    plt.close('all')

    return selectivity_maps





def Plot_Single_Img_Map(single_img_maps, s_dir, e, wandb_on=True, name='Single_Img_Map'):
    s_dim = single_img_maps[0][1].shape[0]
    sq = int(float(s_dim)**0.5)
    n_classes = len(single_img_maps)

    # Make individual plots
    for x, am, class_name in single_img_maps:
        # Make Continuous Plots
        am = am.view(sq, sq)
        plt.figure(figsize=(8,6))
        ax = sns.heatmap(data=am.cpu(),
                        linewidths=0.1,
                        # vmin=-1.0,
                        # vmax=1.0,
                        # center=0.0,
                        linecolor="#363636")
        cbar = ax.collections[0].colorbar
        # cbar.set_ticks([-0.7, 0., 0.7])
        # cbar.set_ticklabels( ['Control', 'None', class_name])
        cbar.ax.tick_params(labelsize=15)
        plt.xticks([])
        plt.yticks([])
        plt.show()

        if wandb_on:
            wandb.log({f'{name}_Act_{class_name}': wandb.Image(plt)})
            wandb.log({f'{name}_Input_{class_name}': wandb.Image(x)})
        else:
            path = os.path.join(s_dir,'{name}_Act_{class_name}.png')
            plt.savefig(path)
        plt.close('all')
