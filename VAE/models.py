import torch
import torch.nn as nn
from torch.nn import functional as F


# Initialise weights of the model with certain mean and standard deviation
def weights_init_general(model, mean, std):
    for m in model._modules:
        if isinstance(m, nn.ConvTranspose2d) or isinstance(m, nn.Conv2d):
            model._modules[m].weight.data.normal_(mean, std)
            model._modules[m].bias.data.zero_()


class VAE(nn.Module):
    def __init__(self, input_size, general_complexity, weights_mean, weights_std, code_size):
        super(VAE, self).__init__()

        self.input_size = input_size

        self.encoder = nn.Sequential(
            nn.Linear(input_size, 8 * general_complexity),
            nn.ReLU(True),
            nn.Linear(8 * general_complexity, 4 * general_complexity),
            nn.ReLU(True),
            nn.Linear(4 * general_complexity, 2 * general_complexity),
            nn.ReLU(True),
            nn.Linear(2 * general_complexity, 1 * general_complexity),
            nn.ReLU(True)
        )

        self.layer_split_1 = nn.Linear(1 * general_complexity, code_size)
        self.layer_split_2 = nn.Linear(1 * general_complexity, code_size)

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

    def encode(self, input):
        output = self.encoder(input)
        return self.layer_split_1(output), self.layer_split_2(output)
    
    def reparametrize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        output = eps.mul(std).add_(mu)
        return output

    def decode(self, latent_vector):
        output = self.decoder(latent_vector)
        return output

    def forward(self, input):
        mu, logvar = self.encode(input)
        latent_vector = self.reparametrize(mu, logvar)
        output = self.decode(latent_vector), mu, logvar
        return output
    
    def loss(self, decoded_output, input, mu, logvar):
        loss = nn.BCELoss(size_average=False)
        bce_loss = loss(decoded_output, input.view(-1, self.input_size))

        # Taken from: https://arxiv.org/abs/1312.6114
        KL_Divergence = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())

        total_loss = bce_loss + KL_Divergence

        return total_loss

    def loss_function(self, recon_x, x, mu, logvar):
        BCE = F.binary_cross_entropy(recon_x, x.view(-1, self.input_size), size_average=False)

        # see Appendix B from VAE paper:
        # Kingma and Welling. Auto-Encoding Variational Bayes. ICLR, 2014
        # https://arxiv.org/abs/1312.6114
        # 0.5 * sum(1 + log(sigma^2) - mu^2 - sigma^2)
        KLD = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())

        return BCE + KLD