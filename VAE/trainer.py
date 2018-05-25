import sys
import os
sys.path.append(os.path.abspath('../utils'))

import utils

import torch
import torch.optim as optim

from torchvision import datasets
import torchvision.transforms as transforms
from torchvision.utils import save_image
from torchvision.utils import make_grid

from hyperparameters import *
from models import VAE as VAE


class VAETrainer():

    def __init__(self, save_path=SAVE_PATH, nb_image_to_gen=NB_IMAGE_TO_GENERATE,
                 code_size=CODE_SIZE, image_size=IMAGE_SIZE, model_complexity=COMPLEXITY, 
                 mean=WEIGHTS_MEAN, std=WEIGHTS_STD, learning_rate=LEARNING_RATE,
                 image_channels=IMAGE_CHANNELS, batch_size=MINIBATCH_SIZE):

        self.batch_size = batch_size
        self.save_path = save_path
        self.nb_image_to_gen = nb_image_to_gen
        self.image_size = image_size
        self.image_channels = image_channels
                
        # Device (cpu or gpu)
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

        # Model
        input_size = image_size * image_size * self.image_channels  # times 3 because of colors
        self.VAE = VAE(input_size, model_complexity, mean, std, code_size).to(self.device)

        self.optimiser = optim.Adam(self.VAE.parameters(), lr = learning_rate)

        self.losses = []

        self.saved_code_input = torch.randn((self.nb_image_to_gen, code_size)).to(self.device)

        # Create directory for the results if it doesn't already exists
        import os
        os.makedirs(self.save_path, exist_ok=True)
        os.makedirs(self.save_path + "encoded/", exist_ok=True)
        os.makedirs(self.save_path + "decoded/", exist_ok=True)
        os.makedirs(self.save_path + "saved_generated/", exist_ok=True)
    
    def load_dataset(self, name):
        print("Loading ", name, " dataset.")
        if name == "MNIST":
            self.train_loader = utils.load_mnist(self.image_size, self.batch_size, root="../MNIST_data")
        elif name == "CIFAR10":
            self.train_loader = utils.load_cifar_10(self.image_size, self.batch_size, root="../CIFAR10_data")
        elif name == "POKEMON":
            self.train_loader = utils.load_pokemon(self.image_size, self.batch_size, root="../POKEMON_data")
        else:
            raise NameError("The only supported datasets are MNIST, CIFAR10 and POKEMON.")

    def train(self, nb_epoch=NB_EPOCH, batch_size=MINIBATCH_SIZE):
        self.VAE.train()
        for epoch in range(nb_epoch):

            print("Epoch : " + str(epoch))
            current_loss = []

            for batch_id, (x, target) in enumerate(self.train_loader):

                current_batch_size = x.shape[0]
                real_batch_data = x.view(current_batch_size, -1).to(self.device)

                self.VAE.zero_grad()

                reconstructed_batch, mu, logvar = self.VAE(real_batch_data)

                loss = self.VAE.loss(reconstructed_batch, real_batch_data, mu, logvar)

                loss.backward()
                self.optimiser.step()

                current_loss.append(loss.item())

                if batch_id == len(self.train_loader) - 2:
                    utils.save_images(real_batch_data, self.save_path + "encoded/", self.image_size, self.image_channels, epoch)
                    utils.save_images(reconstructed_batch, self.save_path + "decoded/", self.image_size, self.image_channels, epoch)

            self.losses.append(torch.mean(torch.tensor(current_loss)))

            utils.save_images(self.VAE.decode(self.saved_code_input), self.save_path + "saved_generated/", self.image_size, self.image_channels, epoch)

            utils.write_loss_plot(self.losses, "loss", self.save_path)

    def save_models(self):
        utils.save_model(self.VAE, self.save_path, "autoencoder_end")
    
    def save_parameters(self):
        utils.save_parameters(self.save_path)