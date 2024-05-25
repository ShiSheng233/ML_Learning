# Config and Hyperparameters
device = 'cuda'
config = {
    'seed': 712,  # Seed for reproducibility.
    'valid_ratio': 0.2,  # validation_size = train_size * valid_ratio
    'n_epochs': 10000,  # Number of epochs.
    'batch_size': 256,
    'learning_rate': 1e-5,
    'early_stop': 400,  # If model has not improved for this many consecutive epochs, stop training.
    'save_path': './models/model-adam.ckpt'
}
