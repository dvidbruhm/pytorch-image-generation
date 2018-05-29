import utils

import torch
import torch.nn as nn

class Generator(nn.Module):
    def __init__(self, latent_size, general_complexity, dropout_prob, weights_mean, weights_std, image_channels, num_labels, image_size, label_latent_input):
        super(Generator, self).__init__()

        self.loss = nn.BCELoss()
        
        self.image_size = image_size
        self.image_channels = image_channels
        self.general_complexity = general_complexity

        self.label_fc_layer = nn.Sequential(
            nn.Linear(num_labels, label_latent_input),
            nn.ReLU(True)
        )
        self.fc_layer1 = nn.Sequential(
            nn.Linear(latent_size + label_latent_input, 4 * image_channels * general_complexity * image_size * image_size),
            nn.ReLU(True)
        )        
        self.layer2 = nn.Sequential(
            nn.ConvTranspose2d(4 * image_channels * general_complexity, 2 * image_channels * general_complexity, 3, 1, 1, bias=False),
            nn.BatchNorm2d(2 * image_channels * general_complexity),
            nn.ReLU(True),
            nn.Dropout2d(dropout_prob)
        )
        self.layer3 = nn.Sequential(
            nn.ConvTranspose2d(2 * image_channels * general_complexity, 1 * image_channels * general_complexity, 3, 1, 1, bias=False),
            nn.BatchNorm2d(1 * image_channels * general_complexity),
            nn.ReLU(True),
            nn.Dropout2d(dropout_prob)
        )        
        self.layer4 = nn.Sequential(
            nn.ConvTranspose2d(1 * image_channels * general_complexity, 1 * image_channels, 3, 1, 1, bias=False),
            nn.Tanh()
        )
        
        utils.weights_init_general(self, weights_mean, weights_std)
    
    def forward(self, input, label):
        current_batch_size = input.shape[0]

        label = self.label_fc_layer(label)

        input = input.view(current_batch_size, -1)
        input = torch.cat([input, label], 1)
        
        output = self.fc_layer1(input)
        output = output.view(current_batch_size, 4 * self.general_complexity * self.image_channels, self.image_size, self.image_size)
        output = self.layer2(output)
        output = self.layer3(output)
        output = self.layer4(output)
        return output


class Discriminator(nn.Module):
    def __init__(self, general_complexity, weights_mean, weights_std, packing, image_channels, num_labels, image_size, label_latent_input):
        super(Discriminator, self).__init__()

        self.loss = nn.BCELoss()

        input_size = image_channels * packing

        self.layer1 = nn.Sequential(
            nn.Conv2d(input_size, general_complexity, 3, 1, 1, bias=False),
            nn.LeakyReLU(0.2, inplace=True)
        )
        self.layer2 = nn.Sequential(
            nn.Conv2d(general_complexity, general_complexity * 2, 3, 1, 1, bias=False),
            nn.BatchNorm2d(general_complexity * 2),
            nn.LeakyReLU(0.2, inplace=True)
        )
        self.layer3 = nn.Sequential(
            nn.Conv2d(general_complexity * 2, general_complexity * 4, 3, 1, 1, bias=False),
            nn.BatchNorm2d(general_complexity * 4),
            nn.LeakyReLU(0.2, inplace=True)
        )
        self.label_fc_layer = nn.Sequential(
            nn.Linear(num_labels, label_latent_input),
            nn.ReLU(True)
        )
        self.fc_layer4 = nn.Sequential(
            nn.Linear(4 * general_complexity * image_size * image_size + label_latent_input, 8 * general_complexity),
            nn.ReLU(True),
            nn.Linear(8 * general_complexity, 1),
            nn.Sigmoid()
        )
    
        utils.weights_init_general(self, weights_mean, weights_std)
            
    def forward(self, input, label):
        current_batch_size = input.shape[0]
        output = self.layer1(input)
        output = self.layer2(output)
        output = self.layer3(output)
        output = output.view(current_batch_size, -1)
        label = self.label_fc_layer(label)
        output = torch.cat([output, label], 1)
        output = self.fc_layer4(output)
        return output
