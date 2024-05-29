import torch.nn as nn
import torch.nn.functional as F
import torch
import numpy as np

from model.block_handler.block_factory import BlockFactory

class Generator(nn.Module):
    def __init__(self, img_shape, latent_dim, block_name):
        super(Generator, self).__init__()
        self.img_shape = img_shape
        self.latent_dim = latent_dim

        self.model = nn.Sequential(
            *BlockFactory.get_block(block_name, self.latent_dim, 128, normalize=True),
            *BlockFactory.get_block(block_name, 128, 256),
            *BlockFactory.get_block(block_name, 256, 512),
            *BlockFactory.get_block(block_name, 512, 1024),
            nn.Linear(1024, int(np.prod(self.img_shape))),
            nn.Tanh()
        )

    def forward(self, z):
        img = self.model(z)
        img = img.view(img.size(0), *self.img_shape)
        return img