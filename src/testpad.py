import pytorch_lightning as pl
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import numpy as np
from torchmetrics.retrieval import RetrievalPrecision, RetrievalRecall
from sklearn.model_selection import train_test_split
from typing import List, Tuple
import random
import torch.nn.functional as F 
from torch.nn.utils.rnn import pad_sequence
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence

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

class RCNN_NextFuture(pl.LightningModule):
    def __init__(
        self, 
        num_items: int,
        embedding_dim: int,
        hidden_size: int,
        num_layers: int,
        dropout: float = 0.1,
        top_k: List[int] = [5, 10, 20],
        conv_out_channels: int = 8, # hn
        horizontal_filter_size: int = 3, # w
        vertical_filter_size: int = 3 # k
    ):
        super().__init__()
        self.save_hyperparameters()
        
        # Item embedding layer
        self.item_embeddings = nn.Embedding(num_items, embedding_dim)
        
        # LSTM layer
        self.lstm = nn.LSTM(
            input_size=embedding_dim,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0
        )
        
        # Horizontal Convolution Layer
        self.horizontal_conv = nn.Conv2d(
            in_channels=1, 
            out_channels=conv_out_channels, 
            kernel_size=(1, horizontal_filter_size)
        )
        
        # Vertical Convolution Layer
        self.vertical_conv = nn.Conv2d(
            in_channels=1, 
            out_channels=1, 
            kernel_size=(vertical_filter_size, 1)
        )
        
        self.vertical_filter_size = vertical_filter_size
        # Final prediction layer
        self.prediction = nn.Sequential(
            nn.Linear(2 * hidden_size + conv_out_channels, num_items),
            nn.Sigmoid()
        )
        
        # Binary Cross Entropy loss with logits
        self.criterion = nn.BCEWithLogitsLoss()
        
        # Initialize metrics using torchmetrics
        self.train_metrics = nn.ModuleDict({
            f'precision@{k}': RetrievalPrecision(top_k=k) for k in top_k
        } | {
            f'recall@{k}': RetrievalRecall(top_k=k) for k in top_k
        })
        
        self.val_metrics = nn.ModuleDict({
            f'precision@{k}': RetrievalPrecision(top_k=k) for k in top_k
        } | {
            f'recall@{k}': RetrievalRecall(top_k=k) for k in top_k
        })
        
        self.test_metrics = nn.ModuleDict({
            f'precision@{k}': RetrievalPrecision(top_k=k) for k in top_k
        } | {
            f'recall@{k}': RetrievalRecall(top_k=k) for k in top_k
        })
        
    def forward(self, x, lengths):
        batch_size, seq_len = x.size()
        
        # Embedding and LSTM forward pass
        embedded = self.item_embeddings(x)  # (batch_size, seq_len, embedding_dim)
        packed_embedded = pack_padded_sequence(embedded, lengths, batch_first=True, enforce_sorted=False)
        packed_lstm_out, _ = self.lstm(packed_embedded)
        lstm_out, _ = pad_packed_sequence(packed_lstm_out, batch_first=True)  # (batch_size, seq_len, hidden_size)
        
        # Horizontal Convolution
        horizontal_input = lstm_out.unsqueeze(1)  # (batch_size, 1, seq_len, hidden_size)
        horizontal_conv_out = F.relu(self.horizontal_conv(horizontal_input))[:, :, -1, :]  # (batch_size, conv_out_channels, new_seq_len)
        horizontal_conv_out = torch.sum(horizontal_conv_out, dim=2)  # (batch_size, conv_out_channels)
        
        # Vertical Convolution
        vertical_input = lstm_out.unsqueeze(1)[:, :, -self.vertical_filter_size:, :]  # (batch_size, 1, k, hidden_size)
        vertical_conv_out = F.relu(self.vertical_conv(vertical_input))  # (batch_size, 1, 1, hidden_size)
        vertical_conv_out = vertical_conv_out.view(batch_size, -1)  # Flatten # (batch_size, hidden_size)
        
        # Concatenate LSTM output with convolution outputs
        last_hidden = lstm_out[:, -1, :]  # (batch_size, hidden_size)
        vertical_conv_out = torch.mul(vertical_conv_out, last_hidden)  # (batch_size, hidden_size)
        
        combined = torch.cat([horizontal_conv_out, last_hidden, vertical_conv_out], dim=1)
        
        # Final prediction
        logits = self.prediction(combined)
        return logits

    def _compute_metrics(self, logits, targets, metrics_dict, prefix=''):
        probs = torch.sigmoid(logits)
        batch_size = logits.size(0)
        
        # Prepare data for retrieval metrics
        indexes = torch.arange(batch_size).repeat_interleave(targets.size(1))
        preds = probs.view(-1)
        target_labels = targets.view(-1)
        
        # Update metrics
        for name, metric in metrics_dict.items():
            metric(preds, target_labels, indexes=indexes)
            self.log(f'{prefix}{name}', metric, prog_bar=True)
    
    def training_step(self, batch, batch_idx):
        sequences, targets, length = batch
        logits = self(sequences, length)
        loss = self.criterion(logits, targets.float())
        self._compute_metrics(logits, targets, self.train_metrics, 'train_')
        self.log('train_loss', loss)
        return loss
    
    def validation_step(self, batch, batch_idx):
        sequences, targets, length = batch
        logits = self(sequences, length)
        loss = self.criterion(logits, targets.float())
        self._compute_metrics(logits, targets, self.val_metrics, 'val_')
        self.log('val_loss', loss)
        return loss
    
    def test_step(self, batch, batch_idx):
        sequences, targets, length = batch
        logits = self(sequences, length)
        loss = self.criterion(logits, targets.float())
        self._compute_metrics(logits, targets, self.test_metrics, 'test_')
        self.log('test_loss', loss)
        return loss
    
    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=0.001)

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

# Example usage
if __name__ == "__main__":

    # Define parameters
    num_sequences = 100  # Number of sequences
    sequence_length = 20  # Length of each sequence
    sequence_start = 1  # Start range
    sequence_end = 49  # End range

    # Generate the sequences
    user_sequences = [
        [random.randint(sequence_start, sequence_end) for _ in range(sequence_length)]
        for _ in range(num_sequences)
    ]

    # Hyperparameters
    NUM_ITEMS = 50
    MIN_SEQUENCE_LENGTH = 3
    FUTURE_WINDOW = 3
    EMBEDDING_DIM = 32
    HIDDEN_SIZE = 64
    NUM_LAYERS = 1
    BATCH_SIZE = 4
    MAX_EPOCHS = 10
    TOP_K = [5, 10, 20]

    # Create data module with specific split ratios
    data_module = RecommendationDataModule(
        user_sequences=user_sequences,
        num_items=NUM_ITEMS,
        sequence_length=MIN_SEQUENCE_LENGTH,
        future_window=FUTURE_WINDOW,
        batch_size=BATCH_SIZE,
        train_ratio=0.7,
        val_ratio=0.2,
        test_ratio=0.1
    )

    # Create model
    model = RCNN_NextFuture(
        num_items=NUM_ITEMS,
        embedding_dim=EMBEDDING_DIM,
        hidden_size=HIDDEN_SIZE,
        num_layers=NUM_LAYERS,
        top_k=TOP_K
    )

    # Create trainer
    trainer = pl.Trainer(
        max_epochs=MAX_EPOCHS,
        accelerator='auto',
        devices=1,
        fast_dev_run=True
    )

    # Train and test model
    trainer.fit(model, data_module)
    trainer.test(model, data_module)