import torch
import torch.nn as nn
import torch.nn.functional as F

class Block():
    '''
    Base class to reprenent a Dataset
    '''

    def __init__(self, in_feat, out_feat, normalize=True):
        self.in_feat = in_feat
        self.out_feat = out_feat
        self.normalize = normalize
        self.layers = None


class Basic(Block):
    def __init__(self, in_feat, out_feat, normalize=True):
        super().__init__(in_feat, out_feat, normalize)
    
        self.layers = [nn.Linear(self.in_feat, self.out_feat)]
        if self.normalize:
            self.layers.append(nn.BatchNorm1d(self.out_feat, 0.8))
        self.layers.append(nn.LeakyReLU(0.2, inplace=True))