#hparam.py

import os

class Hyperparameters:
    
    #* General Hyperparameters
    max_epochs = 20
    min_epochs = 1
    batch_size = 64
    num_workers = int(os.cpu_count()) - 1 #int(os.cpu_count() / 2)
    train_ratio = 0.7
    val_ratio = 0.2
    test_ratio = 0.1
    random_state = 42
    patience = 10
    learning_rate = 0.005
    weight_decay = 0.005 
    lr_patience = None
    lr_factor = None
    
    #* Time-series Hyperparameters
    n = 8 # horizontal_conv_out_channels
    w = 64 # horizontal_filter_size 
    k = 4 # vertical_filter_size
    min_sequence_length = k
    future_window = 10
    embedding_dim = 128 # latentFac
    hidden_size = 512
    num_lstm_layers = 1
    lstm_dropout = 0.1
    
    #* Model Latent Factor
    latentFac = 8
    path_index = 0