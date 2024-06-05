import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F

class Block():
    '''
    Base class to reprenent a Dataset
    '''

    def __init__(self, in_feat, out_feat, normalize=True, last = False):
        self.in_feat = in_feat
        self.out_feat = out_feat
        self.normalize = normalize
        self.last = last
        self.layers = None


class Basic(Block):
    def __init__(self, in_feat, out_feat, normalize, last):
        super().__init__(in_feat, out_feat, normalize, last)
    
        self.layers = [nn.Linear(self.in_feat, self.out_feat)]
        if self.normalize:
            self.layers.append(nn.BatchNorm1d(self.out_feat, 0.8))
        if self.last == False:
            self.layers.append(nn.LeakyReLU(0.2, inplace=True))

class Basic(Block):
    def __init__(self, in_feat, out_feat, normalize, last):
        super().__init__(in_feat, out_feat, normalize, last)
    
        self.layers = [nn.Linear(self.in_feat, self.out_feat)]

        
        if self.normalize:
            self.layers.append(nn.BatchNorm1d(self.out_feat, 0.8))
        if self.last == False:
            self.layers.append(nn.LeakyReLU(0.2, inplace=True))

class CNN(Block):
    def __init__(self, in_feat, out_feat, normalize, last):
        super().__init__(in_feat, out_feat, normalize, last)
    
        self.layers = [nn.ConvTranspose2d(self.in_feat, self.out_feat, kernel_size=4, stride=2, padding=1, bias=False)]
        if self.normalize:
            self.layers.append(nn.BatchNorm1d(self.out_feat, 0.8))
        if self.last == False:
            self.layers.append(nn.LeakyReLU(0.2, inplace=True))


class TransformerBlock(Block):
    def __init__(self, in_feat, out_feat,  nhead, num_layers, dim_feedforward=2048):
        super().__init__(in_feat, out_feat)
        self.layers = []
        encoder_layers = nn.TransformerEncoderLayer(in_feat, nhead, dim_feedforward)

        self.layers.append(PositionalEncoding(in_feat))
        self.layers.append(nn.TransformerEncoder(encoder_layers, num_layers))
        self.layers.append(nn.Linear(self.in_feat, self.out_feat))
        
class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=5000):
        super(PositionalEncoding, self).__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-np.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)
        self.register_buffer('pe', pe)

    def forward(self, x):
        x = x + self.pe[:x.size(0), :]
        return x