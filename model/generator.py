import torch.nn as nn
import torch.nn.functional as F
import numpy as np

from model.block_handler.block_factory import BlockFactory

class Generator(nn.Module):
    def __init__(self, img_shape, latent_dim, block_name):
        super(Generator, self).__init__()
        self.img_shape = img_shape
        self.latent_dim = latent_dim
        if block_name in ['basic', 'Residual', 'transformer']:
            self.model = nn.Sequential(
                *BlockFactory.get_block(block_name, self.latent_dim, 128, normalize=True),
                *BlockFactory.get_block(block_name, 128, 256),
                *BlockFactory.get_block(block_name, 256, 512),
                *BlockFactory.get_block(block_name, 512, 1024),
                *BlockFactory.get_block(block_name, 1024, int(np.prod(self.img_shape)), last = True),
                nn.Tanh()
            )

        elif block_name in [ 'VGG', 'Resnet', 'Dense','InceptionV1','InceptionV2', 'DCNv3']:
            self.channels = self.img_shape[0]
            self.init_size = self.img_shape[1] // 4  # 28 // 4 = 7
            self.model = nn.Sequential(
                nn.Linear(self.latent_dim, 1024 * self.init_size * self.init_size),
                nn.ReLU(inplace=True),
                nn.Unflatten(1, (1024, self.init_size, self.init_size)),  # Reshape to (batch_size, 1024, init_size, init_size)
                nn.ConvTranspose2d(1024, 1024, kernel_size=4, stride=2, padding=1),  # 7x7 -> 14x14
                *BlockFactory.get_block(block_name, 1024, 512, normalize=True),  
                *BlockFactory.get_block(block_name, 512, 256, normalize=True),
                nn.ConvTranspose2d(256, 256, kernel_size=4, stride=2, padding=1),  # 14x14 -> 28x28
                *BlockFactory.get_block(block_name, 256, 128, normalize=True),
                *BlockFactory.get_block(block_name, 128, 64, normalize=True),
                nn.ConvTranspose2d(64, self.channels, kernel_size=3, stride=1, padding=1),
                #*BlockFactory.get_block(block_name, 64, self.channels, normalize=True),
                nn.Tanh()
            )
            
    def forward(self, z):
        img = self.model(z)
        img = img.view(img.size(0), *self.img_shape)
        return img