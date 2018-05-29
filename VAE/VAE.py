import sys
import os
sys.path.append(os.path.abspath('../utils'))

from trainer import VAETrainer
from hyperparameters import DATASET_NAME

def main():

    # Create trainer for the autoencoder
    trainer = VAETrainer()

    # Save the hyperparameters used for this training
    trainer.save_parameters()

    # Load the dataset used for this training
    trainer.load_dataset(DATASET_NAME)

    # Train
    trainer.train()

    # Save the trained autoencoder
    trainer.save_models()


if __name__ == "__main__":
    main()

