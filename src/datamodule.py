import pytorch_lightning as pl
import torch
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
from typing import List
from torch.nn.utils.rnn import pad_sequence
class UserSequencesDataset(Dataset):
    def __init__(self, user_sequences, num_items, min_sequence_length, future_window):
        """
        user_sequences: list of lists, where each inner list contains chronologically ordered item IDs for a user
        num_items: total number of unique items in the dataset
        min_sequence_length: minimum length of sequence required before making predictions
        future_window: number of future interactions to predict
        """
        self.sequences = []
        self.future_items = []
        
        total_sequences = 0
        filtered_sequences = 0
        
        for user_sequence in user_sequences:
            # Skip if sequence is too short
            if len(user_sequence) < min_sequence_length + future_window:
                filtered_sequences += 1
                continue
                
            # Validate item IDs
            if not all(0 <= item < num_items for item in user_sequence):
                filtered_sequences += 1
                continue
            
            # For each user, create progressive sequences
            # Start from min_sequence_length and go until we have enough items left for future_window
            for t in range(min_sequence_length, len(user_sequence) - future_window + 1):
                total_sequences += 1
                
                # Get all items from start until time t
                current_sequence = user_sequence[:t]
                
                # Get future items (next future_window items after t)
                future_items = user_sequence[t:t + future_window]
                
                # Create multi-hot encoding for future items
                future_vector = torch.zeros(num_items)
                future_vector[future_items] = 1
                
                self.sequences.append(current_sequence)
                self.future_items.append(future_vector)
        
        #self.sequences = pad_sequence([torch.LongTensor(seq) for seq in self.sequences], batch_first=True)
        print(f"Total sequences processed: {total_sequences}")
        print(f"Sequences filtered out: {filtered_sequences}")
        print(f"Final sequences kept: {len(self.sequences)}")
    
    def __len__(self):
        return len(self.sequences)
    
    def __getitem__(self, idx):
        return torch.LongTensor(self.sequences[idx]), self.future_items[idx]

class RecommendationDataModule(pl.LightningDataModule):
    def __init__(
        self,
        user_sequences: List[List[int]],
        num_items: int,
        sequence_length: int,
        future_window: int,
        batch_size: int = 32,
        train_ratio: float = 0.7,
        val_ratio: float = 0.2,
        test_ratio: float = 0.1,
        random_state: int = 42
    ):
        super().__init__()
        assert abs(train_ratio + val_ratio + test_ratio - 1.0) < 1e-5, "Split ratios must sum to 1"
        
        self.user_sequences = user_sequences
        self.num_items = num_items
        self.sequence_length = sequence_length
        self.future_window = future_window
        self.batch_size = batch_size
        self.train_ratio = train_ratio
        self.val_ratio = val_ratio
        self.test_ratio = test_ratio
        self.random_state = random_state
    
    def setup(self, stage=None):
        # First split: train and temp (val + test)
        train_sequences, temp_sequences = train_test_split(
            self.user_sequences,
            train_size=self.train_ratio,
            random_state=self.random_state
        )
        
        # Second split: val and test from temp
        val_ratio_adjusted = self.val_ratio / (self.val_ratio + self.test_ratio)
        val_sequences, test_sequences = train_test_split(
            temp_sequences,
            train_size=val_ratio_adjusted,
            random_state=self.random_state
        )
        
        # Create datasets
        self.train_dataset = UserSequencesDataset(
            train_sequences,
            self.num_items,
            self.sequence_length,
            self.future_window
        )
        
        self.val_dataset = UserSequencesDataset(
            val_sequences,
            self.num_items,
            self.sequence_length,
            self.future_window
        )
        
        self.test_dataset = UserSequencesDataset(
            test_sequences,
            self.num_items,
            self.sequence_length,
            self.future_window
        )
    
    def train_dataloader(self):
        return DataLoader(self.train_dataset, batch_size=self.batch_size, shuffle=True, collate_fn=self.pad_collate)
    
    def val_dataloader(self):
        return DataLoader(self.val_dataset, batch_size=self.batch_size, collate_fn=self.pad_collate)
    
    def test_dataloader(self):
        return DataLoader(self.test_dataset, batch_size=self.batch_size, collate_fn=self.pad_collate)
    
    def pad_collate(module, batch):
        (xx, yy) = zip(*batch)
        x_lens = [len(x) for x in xx]

        xx_pad = pad_sequence(xx, batch_first=True, padding_value=0)
        yy_pad = pad_sequence(yy, batch_first=True, padding_value=0)

        return xx_pad, yy_pad, x_lens