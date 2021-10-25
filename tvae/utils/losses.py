import torch

def Spatial_loss(matrix, D):
    x1 = matrix.view(-1, 4096, 1)
    Sv = torch.std(x1, dim=0)
    Xm = x1 - x1.mean(dim=0)
    C = Xm @ Xm.view(-1, 1, 4096)
    C = C.mean(dim=0)
    S = Sv @ Sv.view(1, -1)
    C = C / S
    loss = torch.abs(C - (1.0 / (D + 1)))
    kloss = loss.sum()-loss.trace()
    return kloss