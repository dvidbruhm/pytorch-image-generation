from trainer import DCGANTrainer
from utils import *

def main():
    # Create trainer for the DCGAN
    trainer = DCGANTrainer()

    # Save the hyperparameters used for this training
    save_parameters(trainer.save_path)

    # Load the dataset
    #trainer.load_dataset()
    trainer.test_load_mnist()

    # Start the training process
    trainer.train()

    # Save models
    save_model(trainer.generator, trainer.save_path, "generator_end")
    save_model(trainer.discriminator, trainer.save_path, "discriminator_end")

if __name__ == "__main__":
    main()