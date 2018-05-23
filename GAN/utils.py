import torch

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