from trainer import DCGANTrainer
import utils
from hyperparameters import DATASET_NAME

def main():
    # Create trainer for the DCGAN
    trainer = DCGANTrainer()

    # Save the hyperparameters used for this training
    utils.save_parameters(trainer.save_path)

    # Load the dataset
    trainer.load_dataset(DATASET_NAME)
    #trainer.test_load_mnist()

    # Start the training process
    trainer.train()

    # Save models
    utils.save_model(trainer.generator, trainer.save_path, "generator_end")
    utils.save_model(trainer.discriminator, trainer.save_path, "discriminator_end")

if __name__ == "__main__":
    main()