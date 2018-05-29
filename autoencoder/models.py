import torch
import torch.nn as nn

import utils

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

        utils.weights_init_general(self, weights_mean, weights_std)

    def forward(self, input):
        output = self.encoder(input)
        output = self.decoder(output)
        return output
    
    def generate(self, input):
        output = self.decoder(input)
        return output