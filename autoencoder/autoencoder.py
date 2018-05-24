from trainer import AutoencoderTrainer

def main():

    # Create trainer for the autoencoder
    trainer = AutoencoderTrainer()

    # Save the hyperparameters used for this training
    trainer.save_parameters()

    # Load the dataset used for this training
    trainer.load_dataset()

    # Train
    trainer.train()

    # Save the trained autoencoder
    trainer.save_models()


if __name__ == "__main__":
    main()

