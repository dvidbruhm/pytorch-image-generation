from trainer import DCGANTrainer

def main():
    # Create trainer for the DCGAN
    trainer = DCGANTrainer()

    # Save the hyperparameters used for this training
    trainer.save_parameters()

    # Load the dataset
    trainer.load_dataset()

    # Start the training process
    trainer.train()

    # Save models
    trainer.save_models("end")

if __name__ == "__main__":
    main()