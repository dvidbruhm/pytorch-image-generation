import torch
import torch.nn as nn

import utils

class Generator(nn.Module):
    def __init__(self, input_size, general_complexity, dropout_prob, weights_mean, weights_std, image_channels, image_size):
        super(Generator, self).__init__()

        self.loss = nn.BCELoss()

        self.layer1_64 = nn.Sequential(
            nn.ConvTranspose2d(input_size, 8 * general_complexity, 4, 1, 0, bias=False),
            nn.BatchNorm2d(8 * general_complexity),
            nn.ReLU(True),
            nn.Dropout2d(dropout_prob)
        )
        self.layer2_64 = nn.Sequential(
            nn.ConvTranspose2d(8 * general_complexity, 4 * general_complexity, 4, 2, 1, bias=False),
            nn.BatchNorm2d(4 * general_complexity),
            nn.ReLU(True),
            nn.Dropout2d(dropout_prob)
        )
        self.layer1_32 = nn.Sequential(
            nn.ConvTranspose2d(input_size, 4 * general_complexity, 4, 1, 0, bias=False),
            nn.BatchNorm2d(4 * general_complexity),
            nn.ReLU(True),
            nn.Dropout2d(dropout_prob)
        )
        self.layer3 = nn.Sequential(
            nn.ConvTranspose2d(4 * general_complexity, 2 * general_complexity, 4, 2, 1, bias=False),
            nn.BatchNorm2d(2 * general_complexity),
            nn.ReLU(True),
            nn.Dropout2d(dropout_prob)
        )
        self.layer4 = nn.Sequential(
            nn.ConvTranspose2d(2 * general_complexity, 1 * general_complexity, 4, 2, 1, bias=False),
            nn.BatchNorm2d(1 * general_complexity),
            nn.ReLU(True),
            nn.Dropout2d(dropout_prob)
        )
        self.layer5 = nn.Sequential(
            nn.ConvTranspose2d(1 * general_complexity, 1 * image_channels, 4, 2, 1, bias=False),
            nn.Tanh()
        )

        if image_size == 32:
            self.all_layers = nn.Sequential(
                self.layer1_32,
                self.layer3,
                self.layer4,
                self.layer5
            )
        elif image_size == 64:
            self.all_layers = nn.Sequential(
                self.layer1_64,
                self.layer2_64,
                self.layer3,
                self.layer4,
                self.layer5
            )
        else:
            raise Exception("Only image size of 32 or 64 is supported")

        utils.weights_init_general(self, weights_mean, weights_std)
    
    def forward(self, input):
        output = self.all_layers(input)
        return output

class Discriminator(nn.Module):
    def __init__(self, general_complexity, weights_mean, weights_std, packing, image_channels, image_size):
        super(Discriminator, self).__init__()

        self.loss = nn.BCELoss()

        self.layer1 = nn.Sequential(
            nn.Conv2d(image_channels * packing, general_complexity, 4, 2, 1, bias=False),
            nn.LeakyReLU(0.2, inplace=True)
        )
        self.layer2 = nn.Sequential(
            nn.Conv2d(general_complexity, general_complexity * 2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(general_complexity * 2),
            nn.LeakyReLU(0.2, inplace=True)
        )
        self.layer3 = nn.Sequential(
            nn.Conv2d(general_complexity * 2, general_complexity * 4, 4, 2, 1, bias=False),
            nn.BatchNorm2d(general_complexity * 4),
            nn.LeakyReLU(0.2, inplace=True)
        )
        self.layer4_32 = nn.Sequential(
            nn.Conv2d(general_complexity * 4, 1, 4, 1, 0, bias=False),
            nn.Sigmoid()
        )
        self.layer4_64 = nn.Sequential(
            nn.Conv2d(general_complexity * 4, general_complexity * 8, 4, 2, 1, bias=False),
            nn.BatchNorm2d(general_complexity * 8),
            nn.LeakyReLU(0.2, inplace=True)
        )
        self.layer5_64 = nn.Sequential(
            nn.Conv2d(general_complexity * 8, 1, 4, 1, 0, bias=False),
            nn.Sigmoid()
        )

        if image_size == 32:
            self.all_layers = nn.Sequential(
                self.layer1,
                self.layer2,
                self.layer3,
                self.layer4_32
            )
        elif image_size == 64:
            self.all_layers = nn.Sequential(
                self.layer1,
                self.layer2,
                self.layer3,
                self.layer4_64,
                self.layer5_64
            )
        else:
            raise Exception("Only image size of 32 or 64 is supported")

        utils.weights_init_general(self, weights_mean, weights_std)

    def forward(self, input):
        output = self.all_layers(input)
        return output
