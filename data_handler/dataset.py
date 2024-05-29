from torchvision import datasets
import torchvision.transforms as transforms
from torchvision.utils import save_image
import torch

import os 

class Dataset():
    '''
    Base class to reprenent a Dataset
    '''

    def __init__(self, img_size, batch_size, data_dir):
        self.img_size = img_size
        self.batch_size = batch_size
        self.data_dir = data_dir
        self.dataloader = None
        self.test_dataloader = None

class MNIST(Dataset):
    def __init__(self, img_size, batch_size, data_dir):
        super().__init__(img_size, batch_size, data_dir)
        os.makedirs(self.data_dir+ "/mnist", exist_ok=True)
        self.dataloader = torch.utils.data.DataLoader(
            datasets.MNIST(
                self.data_dir + "/mnist",
                train=True,
                download=True,
                transform=transforms.Compose(
                    [transforms.Resize(self.img_size), transforms.ToTensor(), transforms.Normalize([0.5], [0.5])]
                ),
            ),
            batch_size=self.batch_size,
            shuffle=True,
        )

        self.test_dataloader = torch.utils.data.DataLoader(
            datasets.MNIST(
                self.data_dir + "/mnist",
                train=False,
                download=True,
                transform=transforms.Compose(
                    [transforms.Resize(self.img_size), transforms.ToTensor(), transforms.Normalize([0.5], [0.5])]
                ),
            ),
            batch_size=self.batch_size//4,  # Set batch size > 1 to avoid BatchNorm issues
            shuffle=False,
        )

class CIFAR_100(Dataset):
    def __init__(self, img_size, batch_size, data_dir):
        super().__init__(img_size, batch_size, data_dir)
        os.makedirs(self.data_dir+ "/CIFAR-100", exist_ok=True)
        self.dataloader = torch.utils.data.DataLoader(
            datasets.CIFAR100(
                self.data_dir+ "/CIFAR-100",
                train=True,
                download=True,
                transform=transforms.Compose(
                    [transforms.Resize(self.img_size), transforms.ToTensor(), transforms.Normalize([0.5], [0.5])]
                ),
            ),
            batch_size=self.batch_size,
            shuffle=True,
        )

        self.test_dataloader = torch.utils.data.DataLoader(
            datasets.CIFAR100(
                self.data_dir+ "/CIFAR-100",
                train=False,
                download=True,
                transform=transforms.Compose(
                    [transforms.Resize(self.img_size), transforms.ToTensor(), transforms.Normalize([0.5], [0.5])]
                ),
            ),
            batch_size=self.batch_size//4,  # Set batch size > 1 to avoid BatchNorm issues
            shuffle=False,
        )

class ImageNet(Dataset):
    def __init__(self, img_size, batch_size, data_dir):
        super().__init__(img_size, batch_size, data_dir)
        os.makedirs(self.data_dir+ "/imagenet", exist_ok=True)
        self.dataloader = torch.utils.data.DataLoader(
            datasets.ImageNet(
                self.data_dir+ "/imagenet",
                split='train',
                download=True,
                transform=transforms.Compose(
                    [transforms.Resize(self.img_size), transforms.ToTensor(), transforms.Normalize([0.5], [0.5])]
                ),
            ),
            batch_size=self.batch_size,
            shuffle=True,
        )

        self.test_dataloader = torch.utils.data.DataLoader(
            datasets.ImageNet(
                self.data_dir+ "/imagenet",
                split='val',
                download=True,
                transform=transforms.Compose(
                    [transforms.Resize(self.img_size), transforms.ToTensor(), transforms.Normalize([0.5], [0.5])]
                ),
            ),
            batch_size=self.batch_size//4,  # Set batch size > 1 to avoid BatchNorm issues
            shuffle=False,
        )
