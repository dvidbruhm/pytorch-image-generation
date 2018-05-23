import torch
import torch.nn as nn


# Initialise weights of the model with certain mean and standard deviation
def weights_init_general(model, mean, std):
    for m in model._modules:
        if isinstance(m, nn.ConvTranspose2d) or isinstance(m, nn.Conv2d):
            model._modules[m].weight.data.normal_(mean, std)
            model._modules[m].bias.data.zero_()


class Autoencoder(nn.Module):
    def __init__(self, input_size, general_complexity, weights_mean, weights_std, code_size):
        super(Autoencoder, self).__init__()

        self.loss = nn.MSELoss()

        self.encoder = nn.Sequential(
            nn.Linear(input_size, 8 * general_complexity),
            nn.ReLU(True),
            nn.Linear(8 * general_complexity, 4 * general_complexity),
            nn.ReLU(True),
            nn.Linear(4 * general_complexity, 2 * general_complexity),
            nn.ReLU(True),
            nn.Linear(2 * general_complexity, 1 * general_complexity),
            nn.ReLU(True),
            nn.Linear(1 * general_complexity, code_size)
        )

        self.decoder = nn.Sequential(
            nn.Linear(code_size, 1 * general_complexity),
            nn.ReLU(True),
            nn.Linear(1 * general_complexity, 2 * general_complexity),
            nn.ReLU(True),
            nn.Linear(2 * general_complexity, 4 * general_complexity),
            nn.ReLU(True),
            nn.Linear(4 * general_complexity, 8 * general_complexity),
            nn.ReLU(True),
            nn.Linear(8 * general_complexity, input_size),
            nn.Tanh()
        )

        weights_init_general(self, weights_mean, weights_std)

    def forward(self, input):
        output = self.encoder(input)
        output = self.decoder(output)
        return output
    
    def generate(self, input):
        output = self.decoder(input)
        return output