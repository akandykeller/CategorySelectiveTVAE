from torch import nn
import torch.nn.functional as F

def FC_Encoder(n_in, n_out):
    model = nn.Sequential(
                nn.Conv2d(n_in, n_out*2,
                    kernel_size=1, stride=1, padding=0))
    return model

def FC_Decoder(n_in, n_out):
    model = nn.Sequential(
                nn.ConvTranspose2d(n_in, n_out, 
                    kernel_size=1, stride=1, padding=0))
    return model