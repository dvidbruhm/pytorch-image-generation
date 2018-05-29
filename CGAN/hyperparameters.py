DATASET_NAME = "MNIST" # MNIST, CIFAR10 or POKEMON
SAVE_PATH = "results_" + DATASET_NAME + "/"

# Needs to match dataset
IMAGE_SIZE = 32
IMAGE_CHANNELS = 1
NUMBER_LABELS = 10

NB_EPOCH = 200

COMPLEXITY = 1
MINIBATCH_SIZE = 64
PACKING = 1

LEARNING_RATE = 0.0002

NB_DISCRIMINATOR_STEP = 1

NB_IMAGE_TO_GENERATE = 10
LATENT_INPUT = 100
LABEL_LATENT_INPUT = 100

REAL_LABEL_SMOOTHING = False
FAKE_LABEL_SMOOTHING = False

DROPOUT_PROB = 0.0

BETA1 = 0.5
BETA2 = 0.999

WEIGHTS_MEAN = 0.0
WEIGHTS_STD = 0.02