# PokeGeneration

This repository contains pytorch implementation of various generative models to generate images on multiple datasets.

list of links to models

# Getting started

## Prerequisites

- [Python 3](https://www.python.org/downloads/)
- [Pytorch](https://pytorch.org/) (with torchvision)
- [Matplotlib](https://matplotlib.org/)
  ```
  pip3 install matplotlib
  ```
  
## Usage

Each model resides in a folder of its own. To run a model, first clone the repository:

```
git clone https://github.com/dvidbruhm/PokeGeneration.git
```

Then, while in the folder of the model you wish to run:

```
python3 <model name>.py
```

For example, if you wish to run the DCGAN model, ```cd``` to the DCGAN folder, then:

```
python3 DCGAN.py
```

The results for every epoch will be (by default) in a folder named ```results_<dataset name>/```. For every epoch there will be an example of generated images and a plot of the loss. At the end of the training all models are saved in this folder.

## Customization

Every model folder has a file named ```hyperparameters.py```. If you want to use different parameters for a model, modify any parameters in this file to your liking and rerun the script. The names of the parameters should be mostly self-explanatory. Note that I use a file instead of command line options to modify parameters because I find it more convenient.

# Datasets

All datasets used can be found on [this google drive](https://drive.google.com/open?id=1WpPrdORSTyya1aGTeobGbC8BlOSYlNoC).

### MNIST

### Fashion MNIST

### CIFAR10

### Paintings

### Pokemon

# Models

Here is the detail implementation of each model and the results for each one on every dataset.

## Generative adversarial networks (GAN)

explication gan

### DCGAN

details

#### Results

### LSGAN

details

#### Results

### Conditional GAN (CGAN)

details

#### Results

## Autoencoders

explication autoencoder

### Autoencoder


details

#### Results

### Variational autoencoder (VAE)

details

#### Results

# References
