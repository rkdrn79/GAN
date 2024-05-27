import data_handler.dataset as data


class DatasetFactory:
    def __init__(self):
        pass

    @staticmethod
    def get_dataset(name):
        if name == 'MNIST':
            return data.MNIST()
        elif name == 'CIFAR-100':
            return data.CIFAR_100()
        elif name == 'ImageNet':
            return data.ImageNet()
