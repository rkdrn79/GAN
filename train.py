import os
import numpy as np
import math

import argparse

import torchvision.transforms as transforms
from torchvision.utils import save_image

from torch.utils.data import DataLoader
from torchvision import datasets
from torch.autograd import Variable

import torch.nn as nn
import torch.nn.functional as F
import torch

from torchvision.models import inception_v3
from scipy.stats import entropy
from PIL import Image
import glob

from model.generator import Generator
from model.discriminator import Discriminator
from data_handler.dataset import Dataset

def main():
    parser = argparse.ArgumentParser(description='FACIL - Framework for Analysis of Class Incremental Learning')

    # miscellaneous args
    parser.add_argument('--n_epochs', type=int, default=100,
                        help='EPOCH (default=%(default)s)')
    parser.add_argument('--batch_size', type=int, default=64,
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
    parser.add_argument('--sample_interval', type=int, default=400,
                        help='Interval betwen image samples (default=%(default)s)')
    
    img_shape = (parser.channels, parser.img_size, parser.img_size)

    # Initialize generator and discriminator
    generator = Generator()
    discriminator = Discriminator()

    adversarial_loss = torch.nn.BCELoss()

    cuda = True if torch.cuda.is_available() else False
    if cuda:
        generator.cuda()
        discriminator.cuda()
        adversarial_loss.cuda()

    # Optimizers
    optimizer_G = torch.optim.Adam(generator.parameters(), lr=parser.lr, betas=(parser.b1, parser.b2))
    optimizer_D = torch.optim.Adam(discriminator.parameters(), lr=parser.lr, betas=(parser.b1, parser.b2))

    Tensor = torch.cuda.FloatTensor if cuda else torch.FloatTensor

    # ----------
    #  Training
    # ----------

    for epoch in range(3):
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
            z = Variable(Tensor(np.random.normal(0, 1, (imgs.shape[0], parser.latent_dim))))

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

            print(
                "[Epoch %d/%d] [Batch %d/%d] [D loss: %f] [G loss: %f]"
                % (epoch, parser.n_epochs, i, len(dataloader), d_loss.item(), g_loss.item())
            )

            batches_done = epoch * len(dataloader) + i
            if batches_done % parser.sample_interval == 0:
                save_image(gen_imgs.data[:25], "images/%d.png" % batches_done, nrow=5, normalize=True)


    # ----------
    #  Save Generated Images from Test Data
    # ----------

    # Generate and save images using the test dataset
    for i, (test_imgs, _) in enumerate(test_dataloader):
        test_z = Variable(Tensor(np.random.normal(0, 1, (test_imgs.shape[0], parser.latent_dim))))
        test_gen_imgs = generator(test_z)

        for j in range(test_gen_imgs.size(0)):
            save_image(test_gen_imgs.data[j], "test_images/test_img_%d.png" % (i * test_dataloader.batch_size + j), normalize=True)


    # Load pretrained Inception v3 model
    inception_model = inception_v3(pretrained=True, transform_input=False).eval()
    if cuda:
        inception_model.cuda()

    # Function to get prediction for a single image
    def get_pred(img, model, cuda):
        img = img.unsqueeze(0)  # Add batch dimension
        if cuda:
            img = img.cuda()
        with torch.no_grad():
            pred = model(img)
            return F.softmax(pred, dim=1).cpu().numpy()

    # Function to calculate Inception Score
    def calculate_inception_score(image_paths, model, cuda=True, splits=10):
        preds = []

        transform = transforms.Compose([
            transforms.Resize((299, 299)),
            transforms.ToTensor(),
            transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])  # Normalize for Inception v3
        ])

        for img_path in image_paths:
            img = Image.open(img_path).convert('RGB')  # Convert image to RGB
            img = transform(img)
            pred = get_pred(img, model, cuda)
            preds.append(pred)

        preds = np.concatenate(preds, axis=0)

        # Now compute the mean kl-div
        split_scores = []

        N = len(preds)
        for k in range(splits):
            part = preds[k * (N // splits): (k + 1) * (N // splits), :]
            py = np.mean(part, axis=0)
            scores = []
            for i in range(part.shape[0]):
                pyx = part[i, :]
                scores.append(entropy(pyx, py))
            split_scores.append(np.exp(np.mean(scores)))

        return np.mean(split_scores), np.std(split_scores)

    # Define the path to the directory containing the test images
    test_image_path = 'test_images/'

    # Load test image paths
    test_image_paths = glob.glob(test_image_path + '*.png')

    # Calculate Inception Score
    mean_is, std_is = calculate_inception_score(test_image_paths, inception_model, cuda=cuda, splits=10)
    print("Inception Score: Mean - {}, Std - {}".format(mean_is, std_is))

    


if __name__ == '__main__':

    main()
