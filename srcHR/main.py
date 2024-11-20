import argparse
import os
import torch
from datamodule import RecommendationDataModule
from model import RCNN_NextFuture
from pytorch_lightning.callbacks import LearningRateMonitor, ModelCheckpoint
from pytorch_lightning.loggers import WandbLogger
from pytorch_lightning import Trainer
from hparam import Hyperparameters
import json
#Meow
def train_model(check_mode: bool) -> None:
    message = '-' * 20 + 'Running in check mode:' + '-' * 20 if check_mode else '-' * 20 + 'Running in real mode:' + '-' * 20
    print(message)
    print('Training the RCNN model...')

    # Define paths
    path_list = ['data/all_checkinsdict.json', 'data/gowalladict.json']
    data_dir = path_list[0] #! Change this to the path of the data you want to use
    
    user_sequences = _extract_data_from_json(data_dir)
    num_items = _count_unique_item(data_dir) + 1

    # Initialize the data module
    data_module = RecommendationDataModule(
        user_sequences=user_sequences,
        num_items=num_items,
        min_sequence_length=Hyperparameters.min_sequence_length,
        future_window=Hyperparameters.future_window,
        batch_size=Hyperparameters.batch_size,
        train_ratio=Hyperparameters.train_ratio,
        val_ratio=Hyperparameters.val_ratio,
        test_ratio=Hyperparameters.test_ratio,
        random_state=Hyperparameters.random_state,
        num_workers=Hyperparameters.num_workers
    )
    # Initialize the model
    model = RCNN_NextFuture(
        num_items=num_items,
        embedding_dim=Hyperparameters.embedding_dim,
        hidden_size=Hyperparameters.hidden_size,
        num_layers=Hyperparameters.num_lstm_layers,
        lstm_dropout=Hyperparameters.lstm_dropout,
        top_k=Hyperparameters.top_k,
        conv_out_channels=Hyperparameters.n,
        horizontal_filter_size=Hyperparameters.w,
        vertical_filter_size=Hyperparameters.k,
        learning_rate=Hyperparameters.learning_rate,
        weight_decay=Hyperparameters.weight_decay
    )

    # Set up callbacks and logger
    lr_monitor = LearningRateMonitor(logging_interval='epoch')
    wandb_logger = WandbLogger(project='RCNN-Recommender', save_dir='wandb_log')
    check_point = ModelCheckpoint(monitor='val_loss')

    # Initialize the trainer
    trainer = Trainer(devices='auto',
                      accelerator='auto',
                      max_epochs=Hyperparameters.max_epochs,
                      min_epochs=Hyperparameters.min_epochs,
                      logger= wandb_logger,
                      callbacks=[lr_monitor, check_point],
                      fast_dev_run=check_mode,
                      log_every_n_steps=25,
                      profiler='simple')

    # Train and test the model
    trainer.fit(model, datamodule=data_module)
    if check_mode:
        trainer.test(model, datamodule=data_module)
    else:
        trainer.validate(model, datamodule=data_module, ckpt_path='best')
        trainer.test(model, datamodule=data_module, ckpt_path='best')

def _extract_data_from_json(path: str) -> list[list[int]]:
    # Read dictionary from the JSON file
    with open(path, 'r') as file:
        loaded_dict = json.load(file)
    itemlist=[]
    for i in loaded_dict :
        if(len(loaded_dict[i])<50):
            itemlist.append(loaded_dict[i])
    return itemlist

def _count_unique_item(path: str) -> int:
    uniqueItem=[]
    with open(path, 'r') as file:
        loaded_dict = json.load(file)
    for i in loaded_dict:
        for j in loaded_dict[i]:
            if j not in uniqueItem:
                uniqueItem.append(j)
    return len(uniqueItem)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Train the RCNN model')
    parser.add_argument('--check', action='store_true', help='Run in check mode (fast dev run)')
    args = parser.parse_args()

    train_model(check_mode=args.check)

# train_model(check_mode=True)