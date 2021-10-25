import os
import torch
from torch.utils.data import DataLoader
import torchvision.datasets as sets
import torchvision.transforms as transforms
import numpy as np

def get_mean_std(dir, ratio=0.01):
    mean = [0.48300076, 0.45126104, 0.3998704]
    std = [0.26990137, 0.26078254, 0.27288908]
    return mean, std


class Imagenet_Preprocessor:
    def __init__(self, config, resize=256, crop=224):
        self.train_datadir = config['train_datadir']
        self.test_obj_datadir= config['test_obj_datadir']
        self.test_other_datadir = config['test_other_datadir']
        self.batch_size = config['batch_size']
        self.seed = config['seed']
        self.resize = resize
        self.crop = crop

        print("Computing Training Mean")
        self.mean, self.std = get_mean_std(self.train_datadir, ratio=0.01)

        self.transform = transforms.Compose([
                                            transforms.Resize(resize),
                                            transforms.RandomResizedCrop(crop),
                                            transforms.ToTensor(),
                                            transforms.Normalize(mean=self.mean,
                                                                std=self.std),
                                            ])


    def get_dataloaders(self):
        print("Loading datasets")
        train_set = sets.ImageFolder(self.train_datadir, self.transform)
        test_obj_set = sets.ImageFolder(self.test_obj_datadir, self.transform)
        test_other_set = sets.ImageFolder(self.test_other_datadir, self.transform)

        kwargs = {'num_workers': 50, 'pin_memory': True} if torch.cuda.is_available() else {}
        train_loader = DataLoader(train_set, batch_size=self.batch_size, 
                                      shuffle=True, drop_last=True, **kwargs)
        test_obj_loader = DataLoader(test_obj_set, batch_size=self.batch_size, 
                                     shuffle=False, drop_last=False, **kwargs)
        test_other_loader = DataLoader(test_other_set, batch_size=self.batch_size, 
                                      shuffle=False, drop_last=False, **kwargs)

        return train_loader, test_obj_loader, test_other_loader





class Multiset_Preprocessor:
    def __init__(self, config, resize=256, crop=224):
        self.train_datadir = config['train_datadir']
        self.test_datadirs = config['test_datadirs']
        self.batch_size = config['batch_size']
        self.seed = config['seed']
        self.resize = resize
        self.crop = crop

        print("Computing Training Mean")
        self.mean, self.std = get_mean_std(self.train_datadir, ratio=0.01)

        self.transform = transforms.Compose([
                                            transforms.Resize(resize),
                                            transforms.RandomResizedCrop(crop),
                                            transforms.ToTensor(),
                                            transforms.Normalize(mean=self.mean,
                                                                std=self.std),
                                            ])

        self.test_transform = transforms.Compose([
                                            transforms.Resize(crop),
                                            transforms.RandomResizedCrop(crop),
                                            transforms.ToTensor(),
                                            transforms.Normalize(mean=self.mean,
                                                                std=self.std),
                                            ])


    def get_dataloaders(self):
        print("Loading datasets")
        kwargs = {'num_workers': 50, 'pin_memory': True} if torch.cuda.is_available() else {}

        print(f"Train set: {self.train_datadir}")
        train_set = sets.ImageFolder(self.train_datadir, self.transform)
        train_loader = DataLoader(train_set, batch_size=self.batch_size, 
                                      shuffle=True, drop_last=True, **kwargs)

        test_loaders = []

        for name, dd in self.test_datadirs:
            print(f"Test Set {name}: {dd}")
            test_set = sets.ImageFolder(dd, self.test_transform)
            test_loaders.append(DataLoader(test_set, batch_size=self.batch_size, 
                                      shuffle=False, drop_last=False, **kwargs))

        return train_loader, test_loaders