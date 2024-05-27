import torch
import torch.nn as nn
import torch.nn.functional as F

class Block():
    '''
    Base class to reprenent a Dataset
    '''

    def __init__(self, latent_dim, in_feat, out_feat, normalize=True):
        self.latent_dim = latent_dim
        self.in_feat = in_feat
        self.out_feat = out_feat
        self.normalize = normalize



class Basic(Block):
    def __init__(self):
        super().__init__(128, 64, 64)
    
        layers = [nn.Linear(self.in_feat, self.out_feat)]
        if self.normalize:
            layers.append(nn.BatchNorm1d(self.out_feat, 0.8))
        layers.append(nn.LeakyReLU(0.2, inplace=True))
        return layers