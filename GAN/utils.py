import torch
import matplotlib.pyplot as plt
import matplotlib.image as image

# Function that rescales data in the 0-1 range
def rescale_for_rgb_image(images):
    min_val = images.data.min()
    max_val = images.data.max()
    return (images.data-min_val)/(max_val-min_val)

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

def write_image(image_data, save_path, epoch):
    image_data = rescale_for_rgb_image(image_data)
    image.imsave(save_path + "gen_epoch_" + str(epoch) + ".png", image_data.data)

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
