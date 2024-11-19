import pytorch_lightning as pl
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import numpy as np
from torchmetrics import Metric
from typing import List, Set

class NextFutureDataset(Dataset):
    def __init__(self, user_sequences, num_items, sequence_length, future_window):
        """
        user_sequences: list of lists, where each inner list contains item IDs for a user
        num_items: total number of unique items in the dataset
        sequence_length: number of interactions to consider for prediction
        future_window: number of future interactions to predict
        """
        self.sequences = []
        self.future_items = []
        
        for sequence in user_sequences:
            if len(sequence) < sequence_length + future_window:
                continue
            
            for i in range(len(sequence) - sequence_length - future_window + 1):
                input_seq = sequence[i:i + sequence_length]
                future_seq = sequence[i + sequence_length:i + sequence_length + future_window]
                
                # Convert future items to multi-hot encoding
                future_vector = torch.zeros(num_items)
                future_vector[future_seq] = 1
                
                self.sequences.append(input_seq)
                self.future_items.append(future_vector)
    
    def __len__(self):
        return len(self.sequences)
    
    def __getitem__(self, idx):
        return torch.LongTensor(self.sequences[idx]), self.future_items[idx]

class NextFutureDataModule(pl.LightningDataModule):
    def __init__(
        self,
        user_sequences: List[List[int]],
        num_items: int,
        sequence_length: int,
        future_window: int,
        batch_size: int = 32,
        train_split: float = 0.8
    ):
        super().__init__()
        self.user_sequences = user_sequences
        self.num_items = num_items
        self.sequence_length = sequence_length
        self.future_window = future_window
        self.batch_size = batch_size
        self.train_split = train_split
    
    def setup(self, stage=None):
        # Split users into train and validation
        num_users = len(self.user_sequences)
        train_size = int(num_users * self.train_split)
        
        self.train_sequences = self.user_sequences[:train_size]
        self.val_sequences = self.user_sequences[train_size:]
    
    def train_dataloader(self):
        train_dataset = NextFutureDataset(
            self.train_sequences,
            self.num_items,
            self.sequence_length,
            self.future_window
        )
        return DataLoader(train_dataset, batch_size=self.batch_size, shuffle=True)
    
    def val_dataloader(self):
        val_dataset = NextFutureDataset(
            self.val_sequences,
            self.num_items,
            self.sequence_length,
            self.future_window
        )
        return DataLoader(val_dataset, batch_size=self.batch_size)