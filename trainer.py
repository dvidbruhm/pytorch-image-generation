import torch
import torch.nn
import torch.optim as optim
from hyperparameters import *

from torchvision import datasets
import torchvision.transforms as transforms
from models import Generator, Discriminator

import matplotlib.pyplot as plt
import matplotlib.image as image

class DCGANTrainer():
    def __init__(self, save_path=SAVE_PATH, beta1=BETA1, beta2=BETA2, 
                 nb_image_to_gen=NB_IMAGE_TO_GENERATE, latent_input=LATENT_INPUT,
                 image_size=IMAGE_SIZE, weights_mean=WEIGHTS_MEAN, weights_std=WEIGHTS_STD,
                 model_complexity=COMPLEXITY, learning_rate=LEARNING_RATE):

        self.latent_input = latent_input
        self.nb_image_to_gen = nb_image_to_gen
        self.image_size = image_size
        self.save_path = save_path
        
        # Device (cpu or gpu)
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

        # Models
        self.generator = Generator(latent_input, model_complexity, weights_mean, weights_std).to(self.device)
        self.discriminator = Discriminator(model_complexity, weights_mean, weights_std).to(self.device)

        # Optimizers
        self.D_optimiser = optim.Adam(self.discriminator.parameters(), lr = learning_rate, betas = (beta1, beta2))
        self.G_optimiser = optim.Adam(self.generator.parameters(), lr = learning_rate, betas = (beta1, beta2))

        self.generator_losses = []
        self.discriminator_losses = []

        self.saved_latent_input = torch.randn((nb_image_to_gen, latent_input, 1, 1)).to(self.device)

    def load_dataset(self, path_to_data=DATA_PATH):
        print("Loading dataset in : ", path_to_data)

        transform = transforms.Compose([
                        transforms.ToTensor(),
                        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        ])

        train_set = datasets.ImageFolder(root=path_to_data, transform=transform)

        print('Number of images: ', len(train_set))
        print('Sample image shape: ', train_set[0][0].shape, end='\n\n')
        
        self.train_loader = torch.utils.data.DataLoader(train_set, batch_size=MINIBATCH_SIZE, shuffle=True, num_workers=2)
    

    def train(self, nb_epoch = NB_EPOCH):
        print("Start training.")

        for epoch in range(nb_epoch):

            print("Epoch : " + str(epoch))
            g_loss = []
            d_loss = []

            for batch_id, (x, target) in enumerate(self.train_loader):

                real_batch_data = x.to(self.device)
                current_batch_size = x.shape[0]

                # labels
                label_real = torch.full((current_batch_size,), 1, device=self.device)
                label_fake = torch.full((current_batch_size,), 0, device=self.device)

                # Generate with noise
                latent_noise = torch.randn(current_batch_size, self.latent_input, 1, 1, device=self.device)
                generated_batch = self.generator(latent_noise)

                ### Train discriminator
                self.discriminator.zero_grad()

                # Train on real data
                real_prediction = self.discriminator(real_batch_data).squeeze()
                loss_discriminator_real = self.discriminator.loss(real_prediction, label_real)
                #loss_discriminator_real.backward()

                # Train on fake data
                fake_prediction = self.discriminator(generated_batch.detach()).squeeze()
                loss_discriminator_fake = self.discriminator.loss(fake_prediction, label_fake)
                #loss_discriminator_fake.backward()

                # Add losses
                loss_discriminator_total = loss_discriminator_real + loss_discriminator_fake
                loss_discriminator_total.backward()
                self.D_optimiser.step()

                ### Train generator
                self.generator.zero_grad()
                fake_prediction = self.discriminator(generated_batch).squeeze()
                loss_generator = self.generator.loss(fake_prediction, label_real)
                loss_generator.backward()
                self.G_optimiser.step()

                ### Keep track of losses
                d_loss.append(loss_discriminator_total.item())
                g_loss.append(loss_generator.item())

            self.discriminator_losses.append(torch.mean(torch.tensor(d_loss)))
            self.generator_losses.append(torch.mean(torch.tensor(g_loss)))

            self.write_image(epoch)
            self.write_plots()
        
        self.save_models("end")

    def write_image(self, epoch):
        image_data = self.generator(self.saved_latent_input).permute(0, 2, 3, 1).contiguous().view(self.image_size * self.nb_image_to_gen, self.image_size, 3)
        image_data = rescale_for_rgb_image(image_data)
        image.imsave(self.save_path + "gen_epoch_" + str(epoch) + ".png", image_data.data)

    def write_plots(self):
        # Plot losses
        plt.plot(self.generator_losses, label="G loss")
        plt.plot(self.discriminator_losses, label="D loss")

        plt.legend(loc="best")
        plt.xlabel("Epoch")
        plt.ylabel("Loss")

        plt.savefig(self.save_path + "losses.png")

        plt.clf()
 
    def save_models(self, epoch):
        print("Saving models to : " + self.save_path)
        torch.save(self.discriminator.state_dict(), self.save_path + "discriminator_epoch_" + str(epoch) + ".pt")
        torch.save(self.generator.state_dict(), self.save_path + "generator_epoch_" + str(epoch) + ".pt")

def rescale_for_rgb_image(images):
    # Rescale to 0-1 range
    min_val = images.data.min()
    max_val = images.data.max()
    return (images.data-min_val)/(max_val-min_val)



if __name__ == "__main__":
    trainer = DCGANTrainer()
    trainer.load_dataset()
    trainer.train()