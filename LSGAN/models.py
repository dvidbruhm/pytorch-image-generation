import torch
import torch.nn as nn

# Initialise weights of the model with certain mean and standard deviation
def weights_init_general(model, mean, std):
    for m in model._modules:
        if isinstance(m, nn.ConvTranspose2d) or isinstance(m, nn.Conv2d):
            model._modules[m].weight.data.normal_(mean, std)
            model._modules[m].bias.data.zero_()


class Generator32(nn.Module):
    def __init__(self, input_size, general_complexity, dropout_prob, weights_mean, weights_std, image_channels):
        super(Generator32, self).__init__()

        self.loss = nn.MSELoss()

        self.layer1 = nn.Sequential(
            nn.ConvTranspose2d(input_size, 4 * image_channels * general_complexity, 4, 1, 0, bias=False),
            nn.BatchNorm2d(4 * image_channels * general_complexity),
            nn.ReLU(True),
            nn.Dropout2d(dropout_prob)
        )
        self.layer2 = nn.Sequential(
            nn.ConvTranspose2d(4 * image_channels * general_complexity, 2 * image_channels * general_complexity, 4, 2, 1, bias=False),
            nn.BatchNorm2d(2 * image_channels * general_complexity),
            nn.ReLU(True),
            nn.Dropout2d(dropout_prob)
        )
        self.layer3 = nn.Sequential(
            nn.ConvTranspose2d(2 * image_channels * general_complexity, 1 * image_channels * general_complexity, 4, 2, 1, bias=False),
            nn.BatchNorm2d(1 * image_channels * general_complexity),
            nn.ReLU(True),
            nn.Dropout2d(dropout_prob)
        )
        self.layer4 = nn.Sequential(
            nn.ConvTranspose2d(1 * image_channels * general_complexity, 1 * image_channels, 4, 2, 1, bias=False),
            nn.Tanh()
        )

        self.weights_init(weights_mean, weights_std)
    
    def forward(self, input):
        output = self.layer1(input)
        output = self.layer2(output)
        output = self.layer3(output)
        output = self.layer4(output)
        return output

    def weights_init(self, mean, std):
        weights_init_general(self, mean, std)

class Discriminator32(nn.Module):
    def __init__(self, general_complexity, weights_mean, weights_std, packing, image_channels):
        super(Discriminator32, self).__init__()

        self.loss = nn.MSELoss()

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
        self.layer4 = nn.Sequential(
            nn.Conv2d(general_complexity * 4, 1, 4, 1, 0, bias=False),
            nn.Sigmoid()
        )
    
        self.weights_init(weights_mean, weights_std)

    def forward(self, input):
        output = self.layer1(input)
        output = self.layer2(output)
        output = self.layer3(output)
        output = self.layer4(output)
        return output

    def weights_init(self, mean, std):
        weights_init_general(self, mean, std)

class Generator64(nn.Module):
    def __init__(self, input_size, general_complexity, dropout_prob, weights_mean, weights_std, image_channels):
        super(Generator64, self).__init__()

        self.loss = nn.MSELoss()

        self.layer1 = nn.Sequential(
            nn.ConvTranspose2d(input_size, 8 * image_channels * general_complexity, 4, 1, 0, bias=False),
            nn.BatchNorm2d(8 * image_channels * general_complexity),
            nn.ReLU(True),
            nn.Dropout2d(dropout_prob)
        )
        self.layer2 = nn.Sequential(
            nn.ConvTranspose2d(8 * image_channels * general_complexity, 4 * image_channels * general_complexity, 4, 2, 1, bias=False),
            nn.BatchNorm2d(4 * image_channels * general_complexity),
            nn.ReLU(True),
            nn.Dropout2d(dropout_prob)
        )
        self.layer3 = nn.Sequential(
            nn.ConvTranspose2d(4 * image_channels * general_complexity, 2 * image_channels * general_complexity, 4, 2, 1, bias=False),
            nn.BatchNorm2d(2 * image_channels * general_complexity),
            nn.ReLU(True),
            nn.Dropout2d(dropout_prob)
        )
        self.layer4 = nn.Sequential(
            nn.ConvTranspose2d(2 * image_channels * general_complexity, 1 * image_channels * general_complexity, 4, 2, 1, bias=False),
            nn.BatchNorm2d(1 * image_channels * general_complexity),
            nn.ReLU(True),
            nn.Dropout2d(dropout_prob)
        )
        self.layer5 = nn.Sequential(
            nn.ConvTranspose2d(1 * image_channels * general_complexity, 1 * image_channels, 4, 2, 1, bias=False),
            nn.Tanh()
        )

        self.weights_init(weights_mean, weights_std)
    
    def forward(self, input):
        output = self.layer1(input)
        output = self.layer2(output)
        output = self.layer3(output)
        output = self.layer4(output)
        output = self.layer5(output)
        return output

    def weights_init(self, mean, std):
        weights_init_general(self, mean, std)

class Discriminator64(nn.Module):
    def __init__(self, general_complexity, weights_mean, weights_std, packing):
        super(Discriminator64, self).__init__()

        self.loss = nn.MSELoss()

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
        self.layer4 = nn.Sequential(
            nn.Conv2d(general_complexity * 4, general_complexity * 8, 4, 2, 1, bias=False),
            nn.BatchNorm2d(general_complexity * 8),
            nn.LeakyReLU(0.2, inplace=True)
        )
        self.layer5 = nn.Sequential(
            nn.Conv2d(general_complexity * 8, 1, 4, 1, 0, bias=False),
            nn.Sigmoid()
        )
    
        self.weights_init(weights_mean, weights_std)

    def forward(self, input):
        output = self.layer1(input)
        output = self.layer2(output)
        output = self.layer3(output)
        output = self.layer4(output)
        output = self.layer5(output)
        return output

    def weights_init(self, mean, std):
        weights_init_general(self, mean, std)