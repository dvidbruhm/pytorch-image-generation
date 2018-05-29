import sys
import os
sys.path.append(os.path.abspath('../utils'))

from trainer import DCGANTrainer
from hyperparameters import DATASET_NAME

def main():
    # Create trainer for the DCGAN
    trainer = DCGANTrainer()

    # Save the hyperparameters used for this training
    trainer.save_parameters()

    # Load the dataset
    trainer.load_dataset(DATASET_NAME)

    # Start the training process
    trainer.train()

    # Save models
    trainer.save_models()

if __name__ == "__main__":
    main()