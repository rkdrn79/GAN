import os
import numpy as np

import argparse

from torchvision.utils import save_image
from torch.autograd import Variable

import torch

from tqdm import tqdm

from torchvision.models import inception_v3
import glob

from model.generator import Generator
from model.discriminator import Discriminator
from data_handler.dataset_factory import DatasetFactory
from utils.eval import calculate_inception_score

import wandb

import warnings
warnings.filterwarnings("ignore")

def main():
    parser = argparse.ArgumentParser(description='GAN')

    # miscellaneous args
    parser.add_argument('--n_epochs', type=int, default=100,
                        help='EPOCH (default=%(default)s)')
    parser.add_argument('--batch_size', type=int, default=16,
                        help='Batch size (default=%(default)s)')
    parser.add_argument('--lr', type=float, default=0.0002,
                        help='Learning rate (default=%(default)s)')
    parser.add_argument('--b1', type=float, default=0.5,
                        help='B1 (default=%(default)s)')
    parser.add_argument('--b2', type=float, default=0.999,
                        help='B2 (default=%(default)s)')
    parser.add_argument('--n_cpu', type=int, default=8,
                        help='Number of CPU threads to use during batch generation (default=%(default)s)')
    parser.add_argument('--latent_dim', type=int, default=128,
                        help='Dimensionality of the latent space (default=%(default)s)')
    parser.add_argument('--img_size', type=int, default=28,
                        help='Size of each image dimension (default=%(default)s)')
    parser.add_argument('--channels', type=int, default=1,
                        help='Number of image channels (default=%(default)s)')
    parser.add_argument('--sample_interval', type=int, default=100,
                        help='Interval betwen image samples (default=%(default)s)')
    parser.add_argument('--dataset', type=str, default='MNIST',
                        help='Dataset (default=%(default)s)')
    parser.add_argument('--block', type=str, default='basic',
                        help='Block name (default=%(default)s)')
    parser.add_argument('--eval_dir', type = str, default='/home/mingu/GAN/eval_log',
                        help='Image save path')
    parser.add_argument('--data_dir', type = str, default='/home/mingu/GAN/data',
                        help='Image save path')
    
    arg = parser.parse_args()

    import datetime

    current_time = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")

    wandb.init(project='GAN', name= arg.block + "_" + current_time)

    arg.eval_dir = arg.eval_dir + "/" + current_time
    
    img_shape = (arg.channels, arg.img_size, arg.img_size)

    # dataload
    factory = DatasetFactory()
    dataloader, test_dataloader = factory.get_dataset(arg.dataset, arg.img_size, arg.batch_size, arg.data_dir)
    
    # Initialize generator and discriminator
    generator = Generator(img_shape = img_shape, latent_dim = arg.latent_dim, block_name=arg.block)
    discriminator = Discriminator(img_shape = img_shape)
    
    # Load pretrained Inception v3 model
    inception_model = inception_v3(pretrained=True, transform_input=False).eval()

    adversarial_loss = torch.nn.BCELoss()

    cuda = True if torch.cuda.is_available() else False
    if cuda:
        generator.cuda()
        discriminator.cuda()
        inception_model.cuda()
        adversarial_loss.cuda()
    
    # Optimizers
    optimizer_G = torch.optim.Adam(generator.parameters(), lr=arg.lr, betas=(arg.b1, arg.b2))
    optimizer_D = torch.optim.Adam(discriminator.parameters(), lr=arg.lr, betas=(arg.b1, arg.b2))

    Tensor = torch.cuda.FloatTensor if cuda else torch.FloatTensor

    # ----------
    #  Training
    # ----------

    for epoch in tqdm(range(arg.n_epochs)):

        epoch_dir = os.path.join(arg.eval_dir, f"epoch_{epoch}")
        os.makedirs(epoch_dir, exist_ok=True)
        os.makedirs(epoch_dir + '/train', exist_ok=True)
        os.makedirs(epoch_dir + '/test', exist_ok=True)

        generator.train()
        for i, (imgs, _) in enumerate(dataloader):

            # Adversarial ground truths
            valid = Variable(Tensor(imgs.size(0), 1).fill_(1.0), requires_grad=False)
            fake = Variable(Tensor(imgs.size(0), 1).fill_(0.0), requires_grad=False)

            # Configure input
            real_imgs = Variable(imgs.type(Tensor))

            # -----------------
            #  Train Generator
            # -----------------

            optimizer_G.zero_grad()

            # Sample noise as generator input
            z = Variable(Tensor(np.random.normal(0, 1, (imgs.shape[0], arg.latent_dim))))

            # Generate a batch of images
            gen_imgs = generator(z)

            # Loss measures generator's ability to fool the discriminator
            g_loss = adversarial_loss(discriminator(gen_imgs), valid)

            g_loss.backward()
            optimizer_G.step()

            # ---------------------
            #  Train Discriminator
            # ---------------------

            optimizer_D.zero_grad()

            # Measure discriminator's ability to classify real from generated samples
            real_loss = adversarial_loss(discriminator(real_imgs), valid)
            fake_loss = adversarial_loss(discriminator(gen_imgs.detach()), fake)
            d_loss = (real_loss + fake_loss) / 2

            d_loss.backward()
            optimizer_D.step()

            batches_done = epoch * len(dataloader) + i
            if batches_done % arg.sample_interval == 0:
                print(
                "[Epoch %d/%d] [Batch %d/%d] [D loss: %f] [G loss: %f]"
                % (epoch + 1, arg.n_epochs, i, len(dataloader), d_loss.item(), g_loss.item())
                )
                for i in range(arg.batch_size):
                    save_image(gen_imgs.data[i], epoch_dir+ "/train/%d.png" %  (batches_done + i), nrow=5, normalize=True)


        # ----------
        #  Save Generated Images from Test Data
        # ----------

        # Generate and save images using the test dataset
        for i, (test_imgs, _) in enumerate(test_dataloader):
            test_z = Variable(Tensor(np.random.normal(0, 1, (test_imgs.shape[0], arg.latent_dim))))
            test_gen_imgs = generator(test_z)

            for j in range(test_gen_imgs.size(0)):
                save_image(test_gen_imgs.data[j], epoch_dir + "/test/%d.png" % (i * test_dataloader.batch_size + j), normalize=True)

        # ----------
        #  Eval train Data
        # ----------

        # Define the path to the directory containing the test images
        train_image_path = epoch_dir + '/train/'

        # Load test image paths
        train_image_paths = glob.glob(train_image_path + '*.png')

        # Calculate Inception Score
        mean_is, std_is = calculate_inception_score(train_image_paths, inception_model, cuda=cuda, splits=10)
        print("Train Inception Score: Mean - {}, Std - {}".format(mean_is, std_is))
        wandb.log({"Train Inception Score Mean": mean_is, "Train Inception Score Std": std_is})


        # ----------
        #  Eval Test Data
        # ----------

        # Define the path to the directory containing the test images
        test_image_path = epoch_dir + '/test/'

        # Load test image paths
        test_image_paths = glob.glob(test_image_path + '*.png')

        # Calculate Inception Score
        mean_is, std_is = calculate_inception_score(test_image_paths, inception_model, cuda=cuda, splits=10)
        print("Test Inception Score: Mean - {}, Std - {}".format(mean_is, std_is))
        wandb.log({"Test Inception Score Mean": mean_is, "Test Inception Score Std": std_is})



if __name__ == '__main__':

    main()
