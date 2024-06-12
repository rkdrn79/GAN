# GAN

## Usage
1. Clone this repository:
```bash
git clone https://github.com/rkdrn79/GAN.git
cd GAN
```

2. Install the required dependencies:
```bash
conda create -n GAN python=3.8
conda activate GAN
pip install -r requirements.txt
```

3. Set up Weights & Biases (wandb) for experiment tracking. Sign up for a free account on [Weights & Biases](https://www.wandb.com/) and obtain your API key. Initialize wandb using the API key:
```bash
wandb login YOUR_API_KEY
```

4. Run the main script:

```bash
python train.py
```


## Arguments
- `--n_epochs`: Number of epochs for training (default: 100)
- `--batch_size`: Batch size for training (default: 16)
- `--lr`: Learning rate for optimization (default: 0.0002)
- `--b1`: Beta1 for Adam optimizer (default: 0.5)
- `--b2`: Beta2 for Adam optimizer (default: 0.999)
- `--n_cpu`: Number of CPU threads for batch generation (default: 8)
- `--latent_dim`: Dimensionality of the latent space (default: 128)
- `--img_size`: Size of each image dimension (default: 28)
- `--channels`: Number of image channels (default: 1)
- `--sample_interval`: Interval between image samples (default: 100)
- `--dataset`: Dataset name (default: 'MNIST')
- `--block`: Block name (default: 'basic')
- `--eval_dir`: Directory path for evaluation logs
- `--data_dir`: Directory path for dataset

## Experiment Results
During training, the script logs the Inception Score for both the training and test datasets using Weights & Biases. The Inception Score is a metric used to evaluate the quality of generated images by GANs. The experiment results, including the generated images and Inception Scores, can be tracked on the Weights & Biases dashboard.

## References
- [Generative Adversarial Networks (GANs)](https://en.wikipedia.org/wiki/Generative_adversarial_network)
- [PyTorch Documentation](https://pytorch.org/docs/stable/index.html)
- [Weights & Biases Documentation](https://docs.wandb.com/)
