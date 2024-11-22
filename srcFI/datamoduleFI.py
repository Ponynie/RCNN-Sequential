import pytorch_lightning as pl
import torch
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
from typing import List
from torch.nn.utils.rnn import pad_sequence
#datamoduleFI.py
class UserSequencesDataset(Dataset):
    def __init__(self, user_sequences, num_items, sequence_length):
        """
        user_sequences: list of lists, where each inner list contains item IDs for a user
        num_items: total number of unique items in the dataset
        sequence_length: number of interactions to consider for prediction
        """
        self.sequences = []
        self.next_items = []
        
        for sequence in user_sequences:
            # Skip sequences that are too short
            if len(sequence) <= sequence_length:
                continue
            
            # Slide a fixed-size window over the sequence
            for i in range(len(sequence) - sequence_length):
                input_seq = sequence[i:i + sequence_length]
                next_item = sequence[i + sequence_length]  # Immediate next item
                
                self.sequences.append(input_seq)
                self.next_items.append(next_item)
    
    def __len__(self):
        return len(self.sequences)
    
    def __getitem__(self, idx):
        return torch.LongTensor(self.sequences[idx]), self.next_items[idx]
class RecommendationDataModule(pl.LightningDataModule):
    def __init__(self,
                 user_sequences: List[List[int]],
                 num_items: int,
                 sequence_length: int,
                 batch_size: int = 32,
                 train_ratio: float = 0.7,
                 val_ratio: float = 0.2,
                 test_ratio: float = 0.1,
                 random_state: int = 42,
                 num_workers: int = 1):
        
        super().__init__()
        assert abs(train_ratio + val_ratio + test_ratio - 1.0) < 1e-5, "Split ratios must sum to 1"
        
        self.user_sequences = user_sequences
        self.num_items = num_items
        self.sequence_length = sequence_length
        self.batch_size = batch_size
        self.train_ratio = train_ratio
        self.val_ratio = val_ratio
        self.test_ratio = test_ratio
        self.random_state = random_state
        self.num_workers = num_workers
    
    def setup(self, stage=None):
        if stage == 'fit' or stage is None:
            # First split: train and temp (val + test)
            train_sequences, temp_sequences = train_test_split(
                self.user_sequences,
                train_size=self.train_ratio,
                random_state=self.random_state
            )
            
            # Second split: val and test from temp
            val_ratio_adjusted = self.val_ratio / (self.val_ratio + self.test_ratio)
            val_sequences, self.test_sequences = train_test_split(
                temp_sequences,
                train_size=val_ratio_adjusted,
                random_state=self.random_state
            )
        
            # Create datasets
            self.train_dataset = UserSequencesDataset(
                train_sequences,
                self.num_items,
                self.sequence_length,
            )
        
            self.val_dataset = UserSequencesDataset(
                val_sequences,
                self.num_items,
                self.sequence_length,
            )
        elif stage == 'test':
            self.test_dataset = UserSequencesDataset(
                self.test_sequences,
                self.num_items,
                self.sequence_length,
            )
    
    def train_dataloader(self):
        return DataLoader(self.train_dataset, batch_size=self.batch_size, shuffle=True)
    
    def val_dataloader(self):
        return DataLoader(self.val_dataset, batch_size=self.batch_size)
    
    def test_dataloader(self):
        return DataLoader(self.test_dataset, batch_size=self.batch_size)
    