DATASET_NAME = "MNIST" # MNIST, CIFAR10 or POKEMON
SAVE_PATH = "results_" + DATASET_NAME + "/"

# Must match dataset
IMAGE_SIZE = 32
IMAGE_CHANNELS = 1

NB_EPOCH = 2

COMPLEXITY = 1
MINIBATCH_SIZE = 64
CODE_SIZE = 100

LEARNING_RATE = 0.001

NB_IMAGE_TO_GENERATE = 10

WEIGHTS_MEAN = 0.0
WEIGHTS_STD = 0.02