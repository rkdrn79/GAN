import torch.nn as nn
import torch.nn.functional as F
import numpy as np

from model.block_handler.block_factory import BlockFactory

class Generator(nn.Module):
    def __init__(self, img_shape, latent_dim, block_name):
        super(Generator, self).__init__()
        self.img_shape = img_shape
        self.latent_dim = latent_dim
        if block_name in ['basic']:
            self.model = nn.Sequential(
                *BlockFactory.get_block(block_name, self.latent_dim, 128, normalize=True),
                *BlockFactory.get_block(block_name, 128, 256),
                *BlockFactory.get_block(block_name, 256, 512),
                *BlockFactory.get_block(block_name, 512, 1024),
                *BlockFactory.get_block(block_name, 1024, int(np.prod(self.img_shape)), last = True),
                nn.Tanh()
            )

        elif block_name in ['cnn']:
            self.channels = self.img_shape[0]
            self.init_size = self.img_shape[1] // 16
            self.model = nn.Sequential(
                #Fully connected layer to reshape latent vector into a 512 * init_size * init_size feature map
                nn.Linear(self.latent_dim, 1024 * self.init_size * self.init_size),
                nn.ReLU(inplace=True),
                nn.Unflatten(1, (1024, self.init_size, self.init_size)),  # Reshape to (batch_size, 512, init_size, init_size)
                *BlockFactory.get_block(block_name, 1024, 512),
                *BlockFactory.get_block(block_name, 512, 256),
                *BlockFactory.get_block(block_name, 256, 128),
                *BlockFactory.get_block(block_name, 128, 64),
                *BlockFactory.get_block(block_name, 64, self.channels, last = True),
                nn.Tanh()
            )
            

    def forward(self, z):
        img = self.model(z)
        img = img.view(img.size(0), *self.img_shape)
        return img