import utils

import torch
import torch.nn
import torch.optim as optim

from torchvision import datasets
import torchvision.transforms as transforms

from hyperparameters import *
from models import Generator, Discriminator

class CGANTrainer():
    def __init__(self, save_path=SAVE_PATH, beta1=BETA1, beta2=BETA2, 
                 nb_image_to_gen=NB_IMAGE_TO_GENERATE, latent_input=LATENT_INPUT,
                 image_size=IMAGE_SIZE, weights_mean=WEIGHTS_MEAN, weights_std=WEIGHTS_STD,
                 model_complexity=COMPLEXITY, learning_rate=LEARNING_RATE, packing=PACKING,
                 real_label_smoothing=REAL_LABEL_SMOOTHING, fake_label_smoothing=FAKE_LABEL_SMOOTHING,
                 dropout_prob=DROPOUT_PROB, nb_discriminator_step=NB_DISCRIMINATOR_STEP, 
                 image_channels=IMAGE_CHANNELS, batch_size=MINIBATCH_SIZE, num_labels=NUMBER_LABELS,
                 label_latent_input=LABEL_LATENT_INPUT):

        self.num_labels = num_labels
        self.batch_size = batch_size
        self.latent_input = latent_input
        self.nb_image_to_gen = nb_image_to_gen
        self.image_size = image_size
        self.image_channels = image_channels
        self.save_path = save_path
        self.packing = packing
        self.real_label_smoothing = real_label_smoothing
        self.fake_label_smoothing = fake_label_smoothing
        self.nb_discriminator_step = nb_discriminator_step
        
        # Device (cpu or gpu)
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

        # Models
        self.generator = Generator(latent_input, model_complexity, dropout_prob, weights_mean, weights_std, image_channels, num_labels, image_size, label_latent_input).to(self.device)
        self.discriminator = Discriminator(model_complexity, weights_mean, weights_std, packing, image_channels, num_labels, image_size, label_latent_input).to(self.device)

        # Optimizers
        self.D_optimiser = optim.Adam(self.discriminator.parameters(), lr = learning_rate, betas = (beta1, beta2))
        self.G_optimiser = optim.Adam(self.generator.parameters(), lr = learning_rate, betas = (beta1, beta2))

        self.generator_losses = []
        self.discriminator_losses = []

        self.saved_latent_input = torch.randn((nb_image_to_gen * nb_image_to_gen, latent_input, 1, 1)).to(self.device)
        self.saved_label = torch.zeros((nb_image_to_gen * nb_image_to_gen, num_labels))
        for i in range(nb_image_to_gen):
            self.saved_label[i*nb_image_to_gen:i*nb_image_to_gen+nb_image_to_gen,i] = 1

        # Create directory for the results if it doesn't already exists
        import os
        os.makedirs(self.save_path, exist_ok=True)

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

    def train(self, nb_epoch = NB_EPOCH):
        print("Start training.")

        for epoch in range(nb_epoch):

            print("Epoch : " + str(epoch))
            g_loss = []
            d_loss = []

            for batch_id, (x, target) in enumerate(self.train_loader):

                real_batch_data = x.to(self.device)
                current_batch_size = x.shape[0]
                one_hot_label = torch.zeros(current_batch_size, self.num_labels).scatter_(1, target.unsqueeze(1), 1)

                packed_real_data = utils.pack(real_batch_data, self.packing)
                packed_batch_size = packed_real_data.shape[0]

                # labels
                label_real = torch.full((packed_batch_size,), 1, device=self.device)
                label_fake = torch.full((packed_batch_size,), 0, device=self.device)
                # smoothed real labels between 0.7 and 1, and fake between 0 and 0.3
                label_real_smooth = torch.rand((packed_batch_size,)).to(self.device) * 0.3 + 0.7
                label_fake_smooth = torch.rand((packed_batch_size,)).to(self.device) * 0.3

                ### Train discriminator multiple times
                for i in range(self.nb_discriminator_step):
                    loss_discriminator_total = self.train_discriminator(packed_real_data, 
                                                        current_batch_size,
                                                        one_hot_label,
                                                        label_real_smooth if self.real_label_smoothing else label_real,
                                                        label_fake_smooth if self.fake_label_smoothing else label_fake)

                ### Train generator
                loss_generator = self.train_generator(current_batch_size, one_hot_label, label_real)

                ### Keep track of losses
                d_loss.append(loss_discriminator_total.item())
                g_loss.append(loss_generator.item())

            self.discriminator_losses.append(torch.mean(torch.tensor(d_loss)))
            self.generator_losses.append(torch.mean(torch.tensor(g_loss)))

            utils.save_images(self.generator(self.saved_latent_input, self.saved_label), self.save_path + "gen_", self.image_size, self.image_channels, self.nb_image_to_gen, epoch)

            utils.write_loss_plot(self.generator_losses, "G loss", self.save_path, clear_plot=False)
            utils.write_loss_plot(self.discriminator_losses, "D loss", self.save_path, clear_plot=True)

        print("Training finished.")
        

    def train_discriminator(self, real_data, current_batch_size, one_hot_label, real_label, fake_label):
        
        # Generate with noise
        latent_noise = torch.randn(current_batch_size, self.latent_input, 1, 1, device=self.device)
        generated_batch = self.generator(latent_noise, one_hot_label)
        fake_data = utils.pack(generated_batch, self.packing)

        ### Train discriminator
        self.discriminator.zero_grad()

        # Train on real data
        real_prediction = self.discriminator(real_data, one_hot_label).squeeze()
        loss_discriminator_real = self.discriminator.loss(real_prediction, real_label)
        #loss_discriminator_real.backward()

        # Train on fake data
        fake_prediction = self.discriminator(fake_data.detach(), one_hot_label).squeeze()
        loss_discriminator_fake = self.discriminator.loss(fake_prediction, fake_label)
        #loss_discriminator_fake.backward()

        # Add losses
        loss_discriminator_total = loss_discriminator_real + loss_discriminator_fake
        loss_discriminator_total.backward()
        self.D_optimiser.step()

        return loss_discriminator_total


    def train_generator(self, current_batch_size, one_hot_label, real_label):

        # Generate with noise
        latent_noise = torch.randn(current_batch_size, self.latent_input, 1, 1, device=self.device)
        generated_batch = self.generator(latent_noise, one_hot_label)
        fake_data = utils.pack(generated_batch, self.packing)

        ### Train generator
        self.generator.zero_grad()

        fake_prediction = self.discriminator(fake_data, one_hot_label).squeeze()

        # Loss
        loss_generator = self.generator.loss(fake_prediction, real_label)
        loss_generator.backward()
        self.G_optimiser.step()

        return loss_generator

    def save_models(self):
        utils.save_model(self.generator, self.save_path, "generator_end")
        utils.save_model(self.discriminator, self.save_path, "discriminator_end")
    
    def save_parameters(self):
        utils.save_parameters(self.save_path)


