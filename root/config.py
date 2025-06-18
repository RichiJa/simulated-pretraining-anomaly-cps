# config.py

defaults = {
    "window_size": 75,
    "step_size": 25,
    "latent_dim": 256,
    "dropout": 0.18026464026944247,
    "dropout_PT": 0.1335,
    "lr": 0.0009919384966327212,
    "lr_PT": 0.0009606,
    "weight_decay": 3.460251383905326e-05,
    "weight_decay_PT": 7.13e-08,
    "std_factor": 1.5526193433468545,
    "std_factor_PT": 2.8333,
    "batch_size": 32,
    "real_epochs": 17
}
defaultsVAE = {
    # VAE Optuna best trial:
    "window_size": 75,
    "step_size": 25,
    "latent_dim": 32,
    "dropout": 0.1097,
    "lr": 0.0009616,
    "weight_decay": 1.073e-08,
    "std_factor": 1.01,
    "batch_size": 64,
    "real_epochs": 27,
    "sim_epochs": 20
}


data_paths = {
    "real_data_folder": "_",
    "anom_folder_A1": "_",
    "anom_folder_A2": "_",
    "sim_data_folder": "_"
    #unpack included zip file for data and adjust filepaths
}
