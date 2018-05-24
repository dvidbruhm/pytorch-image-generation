from trainer import VAETrainer

def main():

    # Create trainer for the autoencoder
    trainer = VAETrainer()

    # Save the hyperparameters used for this training
    trainer.save_parameters()

    # Load the dataset used for this training
    trainer.load_dataset()
    #trainer.test_load_mnist()

    # Train
    trainer.train()

    # Save the trained autoencoder
    trainer.save_models()


if __name__ == "__main__":
    main()

