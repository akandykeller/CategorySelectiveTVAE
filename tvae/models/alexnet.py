import torch
import torch.nn as nn
import torch.nn.functional as F


def create_alexnet_classifier():
    alexnet = torch.hub.load('pytorch/vision:v0.6.0', 'alexnet', pretrained=True)
    return Alexnet_Classifier(alexnet)

def create_alexnet_full():
    alexnet = torch.hub.load('pytorch/vision:v0.6.0', 'alexnet', pretrained=True)
    return Feature_Extractor_Full(alexnet)

def create_alexnet_fc6():
    alexnet = torch.hub.load('pytorch/vision:v0.6.0', 'alexnet', pretrained=True)
    return Feature_Extractor_PreFC6(alexnet)

class Feature_Extractor_PreFC6(nn.Module):
    def __init__(self, alexnet):
        super(Feature_Extractor_PreFC6, self).__init__()
        # Block 0: input to maxpool1
        self.block0 = nn.Sequential(
            alexnet.features,
            alexnet.avgpool,
        )
        self.fc6 = nn.Sequential(
            nn.BatchNorm1d(9216)
        )

    def forward(self, inp):
        outp = self.block0(inp)
        outp = torch.flatten(outp, 1)
        outp = self.fc6(outp)

        return outp


class Feature_Extractor_Full(nn.Module):
    def __init__(self, alexnet):
        super(Feature_Extractor_Full, self).__init__()
        # Block 0: input to maxpool1
        self.block0 = nn.Sequential(
            alexnet.features,
            alexnet.avgpool,
        )
        self.fc6 = nn.Sequential(
            list(alexnet.classifier)[1],
        )

    def forward(self, inp):
        outp = self.block0(inp)
        outp = torch.flatten(outp, 1)
        outp = self.fc6(outp)
        return outp


class Alexnet_Classifier(nn.Module):
    def __init__(self, alexnet):
        super(Alexnet_Classifier, self).__init__()
        # Block 0: input to maxpool1
        self.block0 = nn.Sequential(
            alexnet.features,
            alexnet.avgpool,
        )
        self.fc6 = nn.Sequential(nn.Dropout(),
            nn.Linear(256 * 6 * 6, 4096))

        self.fc7 = nn.Sequential(nn.Dropout(),
                                 nn.Linear(4096, 4096))

        self.fc8 = nn.Linear(4096, 1001)

    def forward(self, inp):
        outp = self.block0(inp).detach()
        outp = torch.flatten(outp, 1)
        fc6 = self.fc6(outp)
        fc6_rect = F.relu(fc6)
        fc7 = self.fc7(fc6_rect)
        fc7_rect = F.relu(fc7)
        fc8 = self.fc8(fc7_rect)
        return fc8, fc6, fc7