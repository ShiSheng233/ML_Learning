# Config and Hyperparameters
device = 'cuda'
config = {
    'seed': 712,
    'valid_ratio': 0.2,  # validation_size = train_size * valid_ratio
    'n_epochs': 1000,
    'batch_size': 128,
    'learning_rate': 1e-5,
    'save_path': './models/model.ckpt',
    'test_after_training': True,
}
