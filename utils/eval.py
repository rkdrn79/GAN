import os
import numpy as np

import torchvision.transforms as transforms

import torch.nn.functional as F
import torch

from scipy.stats import entropy
from PIL import Image

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