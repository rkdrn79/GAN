from torchvision import datasets
import torchvision.transforms as transforms
from torchvision.utils import save_image
import torch

class Dataset():
    '''
    Base class to reprenent a Dataset
    '''

    def __init__(self, name, img_size, batch_size):
        self.name = name
        self.img_size = img_size
        self.batch_size = batch_size

class MNIST(Dataset):
    def __init__(self):
        super().__init__("MNIST", 28, 64)
        
        dataloader = torch.utils.data.DataLoader(
            datasets.MNIST(
                "../../data/mnist",
                train=True,
                download=True,
                transform=transforms.Compose(
                    [transforms.Resize(self.img_size), transforms.ToTensor(), transforms.Normalize([0.5], [0.5])]
                ),
            ),
            batch_size=self.batch_size,
            shuffle=True,
        )

        # Configure test data loader with batch size of 32 (or any value > 1)
        test_dataloader = torch.utils.data.DataLoader(
            datasets.MNIST(
                "../../data/mnist",
                train=False,
                download=True,
                transform=transforms.Compose(
                    [transforms.Resize(self.img_size), transforms.ToTensor(), transforms.Normalize([0.5], [0.5])]
                ),
            ),
            batch_size=self.batch_size//4,  # Set batch size > 1 to avoid BatchNorm issues
            shuffle=False,
        )

        return dataloader, test_dataloader
    
class CIFAR_100(Dataset):
    def __init__(self):
        super().__init__("CIFAR-100", 32, 64)
        
        dataloader = torch.utils.data.DataLoader(
            datasets.CIFAR100(
                "../../data/cifar-100",
                train=True,
                download=True,
                transform=transforms.Compose(
                    [transforms.Resize(self.img_size), transforms.ToTensor(), transforms.Normalize([0.5], [0.5])]
                ),
            ),
            batch_size=self.batch_size,
            shuffle=True,
        )

        test_dataloader = torch.utils.data.DataLoader(
            datasets.CIFAR100(
                "../../data/cifar-100",
                train=False,
                download=True,
                transform=transforms.Compose(
                    [transforms.Resize(self.img_size), transforms.ToTensor(), transforms.Normalize([0.5], [0.5])]
                ),
            ),
            batch_size=self.batch_size//4,  # Set batch size > 1 to avoid BatchNorm issues
            shuffle=False,
        )

        return dataloader, test_dataloader

class ImageNet(Dataset):
    def __init__(self):
        super().__init__("ImageNet", 64, 64)
        
        dataloader = torch.utils.data.DataLoader(
            datasets.ImageNet(
                "../../data/imagenet",
                split='train',
                download=True,
                transform=transforms.Compose(
                    [transforms.Resize(self.img_size), transforms.ToTensor(), transforms.Normalize([0.5], [0.5])]
                ),
            ),
            batch_size=self.batch_size,
            shuffle=True,
        )

        test_dataloader = torch.utils.data.DataLoader(
            datasets.ImageNet(
                "../../data/imagenet",
                split='val',
                download=True,
                transform=transforms.Compose(
                    [transforms.Resize(self.img_size), transforms.ToTensor(), transforms.Normalize([0.5], [0.5])]
                ),
            ),
            batch_size=self.batch_size//4,  # Set batch size > 1 to avoid BatchNorm issues
            shuffle=False,
        )

        return dataloader, test_dataloader