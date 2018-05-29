import torch
import torch.nn as nn

import matplotlib.pyplot as plt

from torchvision.utils import save_image
from torchvision.utils import make_grid

from torchvision import datasets
import torchvision.transforms as transforms

# Function that takes a minibatch (input) and a packing number (packing)
# and outputs the packed minibatch. 
# See https://arxiv.org/pdf/1712.04086.pdf for more details.
def pack(input, packing):
    # Number of elements that need to be added to the input tensor
    nb_to_add = (packing - (input.shape[0] % packing)) % packing

    # Add elements to the input if not a round number for the packing number
    if nb_to_add > 0:
        input = torch.cat((input, input[-nb_to_add:].view(nb_to_add, 3, input.shape[2], input.shape[3])))

    # Reshape the tensor so it is packed
    packed_output = input.view(-1, input.shape[1] * packing, input.shape[2], input.shape[3])
    return packed_output

# Initialise weights of the model with certain mean and standard deviation
def weights_init_general(model, mean, std):
    for m in model._modules:
        if isinstance(m, nn.ConvTranspose2d) or isinstance(m, nn.Conv2d):
            model._modules[m].weight.data.normal_(mean, std)
            model._modules[m].bias.data.zero_()

def write_loss_plot(loss, loss_label, save_path, clear_plot=True):
    # Plot losses
    plt.plot(loss, label=loss_label)

    plt.legend(loc="best")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")

    plt.savefig(save_path + "losses.png")

    if clear_plot:
        plt.clf()

def save_model(model, save_path, name):
    print("Saving " + name + " to : " + save_path)
    torch.save(model.state_dict(), save_path + name + ".pt")

def save_parameters(save_path, file_name="hyperparameters.py"):
    from shutil import copyfile
    copyfile("hyperparameters.py", save_path + "" + file_name)

def rescale_for_rgb_plot(images):
    min_val = images.data.min()
    max_val = images.data.max()
    return (images.data-min_val)/(max_val-min_val)

def save_images(data, save_path, image_size, image_channels, epoch):
    image_list = []
    for i in range(len(data)):
        image_data = data[i].view(image_channels, image_size, image_size)
        image_data = rescale_for_rgb_plot(image_data)
        image_list.append(image_data)
    save_image(make_grid(image_list), save_path + "epoch_" + str(epoch) + ".png")

def load_cifar_10(image_size=32, batch_size=128, root="../CIFAR10_data"):
    # Create transform
    trans = transforms.Compose([
            transforms.Resize(image_size),
            transforms.ToTensor(),
            transforms.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5))
    ])

    # Load dataset
    train_set = datasets.CIFAR10(root=root, train=True, transform=trans, download=True)

    print('Number of images: ', len(train_set))
    print('Sample image shape: ', train_set[0][0].shape, end='\n\n')
    
    train_loader = torch.utils.data.DataLoader(
                        dataset=train_set,
                        batch_size=batch_size,
                        shuffle=True)

    return train_loader

def load_mnist(image_size=32, batch_size=128, root="../MNIST_data"):
    # Create transform
    trans = transforms.Compose([
            transforms.Resize(image_size),
            transforms.ToTensor(),
            transforms.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5))
    ])

    # Load dataset
    train_set = datasets.MNIST(root=root, train=True, transform=trans, download=True)

    print('Number of images: ', len(train_set))
    print('Sample image shape: ', train_set[0][0].shape, end='\n\n')
    
    train_loader = torch.utils.data.DataLoader(
                        dataset=train_set,
                        batch_size=batch_size,
                        shuffle=True)

    return train_loader

def load_pokemon(image_size=32, batch_size=128, root="../pokemon_data"):

    transform = transforms.Compose([
                    transforms.ToTensor(),
                    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])

    train_set = datasets.ImageFolder(root=root, transform=transform)

    print('Number of images: ', len(train_set))
    print('Sample image shape: ', train_set[0][0].shape, end='\n\n')
    
    train_loader = torch.utils.data.DataLoader(train_set, batch_size=batch_size, shuffle=True, num_workers=2)

    return train_loader
    