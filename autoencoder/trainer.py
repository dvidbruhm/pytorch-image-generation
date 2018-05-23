import torch
import torch.optim as optim

from torchvision import datasets
import torchvision.transforms as transforms
from torchvision.utils import save_image

from hyperparameters import *
from utils import *
from models import Autoencoder


class AutoencoderTrainer():

    def __init__(self, save_path=SAVE_PATH, nb_image_to_gen=NB_IMAGE_TO_GENERATE,
                 code_size=CODE_SIZE, image_size=IMAGE_SIZE, model_complexity=COMPLEXITY, 
                 mean=WEIGHTS_MEAN, std=WEIGHTS_STD, learning_rate=LEARNING_RATE):

        self.save_path = save_path
        self.nb_image_to_gen = nb_image_to_gen
        self.image_size = image_size
                
        # Device (cpu or gpu)
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

        # Model
        input_size = image_size * image_size * 3  # times 3 because of colors
        self.autoencoder = Autoencoder(input_size, model_complexity, mean, std, code_size).to(self.device)

        self.optimiser = optim.Adam(self.autoencoder.parameters(), lr = learning_rate)

        self.losses = []

        self.saved_code_input = torch.randn((nb_image_to_gen, code_size)).to(self.device)

        # Create directory for the results if it doesn't already exists
        import os
        os.makedirs(self.save_path, exist_ok=True)
    
    def load_dataset(self, path_to_data=DATA_PATH, batch_size=MINIBATCH_SIZE):
        print("Loading dataset in : ", path_to_data)

        transform = transforms.Compose([
                        transforms.ToTensor(),
                        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        ])

        train_set = datasets.ImageFolder(root=path_to_data, transform=transform)

        print('Number of images: ', len(train_set))
        print('Sample image shape: ', train_set[0][0].shape, end='\n\n')
        
        self.train_loader = torch.utils.data.DataLoader(train_set, batch_size=batch_size, shuffle=True, num_workers=2)
    
    
    def train(self, nb_epoch=NB_EPOCH, batch_size=MINIBATCH_SIZE):
        for epoch in range(nb_epoch):

            print("Epoch : " + str(epoch))
            current_loss = []

            for batch_id, (x, target) in enumerate(self.train_loader):

                current_batch_size = x.shape[0]
                real_batch_data = x.view(current_batch_size, -1).to(self.device)

                self.autoencoder.zero_grad()

                output = self.autoencoder(real_batch_data)
                #save_image(real_batch_data[0].view(3, self.image_size, self.image_size), self.save_path + "data_epoch_" + str(epoch) + ".png")
                #save_image(output[0].view(3, self.image_size, self.image_size), self.save_path + "gen_epoch_" + str(epoch) + ".png")
                loss = self.autoencoder.loss(output, real_batch_data)

                loss.backward()
                self.optimiser.step()

                current_loss.append(loss.item())
            
            self.losses.append(torch.mean(torch.tensor(current_loss)))

            image_data = self.autoencoder.generate(self.saved_code_input)   \
                             .view(3, self.image_size * self.nb_image_to_gen, self.image_size)

            save_image(image_data, self.save_path + "gen_epoch_" + str(epoch) + ".png")

            write_loss_plot(self.losses, "loss", self.save_path)

                