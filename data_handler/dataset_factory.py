import data_handler.dataset as data


class DatasetFactory:
    def __init__(self):
        pass

    @staticmethod
    def get_dataset(name, img_size, batch_size, data_dir):
        if name == 'MNIST':
            dataset = data.MNIST(img_size, batch_size, data_dir)
            return dataset.dataloader, dataset.test_dataloader
        elif name == 'CIFAR-100':
            dataset = data.CIFAR_100(img_size, batch_size, data_dir)
            return dataset.dataloader, dataset.test_dataloader
        elif name == 'ImageNet':
            dataset = data.ImageNet(img_size, batch_size, data_dir)
            return dataset.dataloader, dataset.test_dataloader
