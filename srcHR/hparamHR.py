#hparam.py

import os

class Hyperparameters:
    
    #* General Hyperparameters
    max_epochs = 100
    min_epochs = 1
    batch_size = 16
    num_workers = 1,# int(os.cpu_count()) - 1 #int(os.cpu_count() / 2)
    train_ratio = 0.7
    val_ratio = 0.2
    test_ratio = 0.1
    random_state = 42
    patience = 10
    learning_rate = 0.01
    weight_decay = 0.001 
    lr_patience = None
    lr_factor = None
    
    #* Time-series Hyperparameters
    n = 8 # horizontal_conv_out_channels
    w = 64 # horizontal_filter_size
    k = 4 # vertical_filter_size
    min_sequence_length = k
    future_window = 10
    embedding_dim = 128
    hidden_size = 256
    num_lstm_layers = 1
    lstm_dropout = 0.1