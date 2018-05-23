from trainer import AutoencoderTrainer
from utils import *

def main():

    # Create trainer for the autoencoder
    trainer = AutoencoderTrainer()

    # Save the hyperparameters used for this training
    save_parameters(trainer.save_path)

    # Load the dataset used for this training
    trainer.load_dataset()

    # Train
    trainer.train()

    # Save the trained autoencoder
    save_model(trainer.autoencoder, trainer.save_path, "autoencoder_end")


if __name__ == "__main__":
    main()

