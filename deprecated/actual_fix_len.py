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

class UserSequencesDataset(Dataset):
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
        
    def forward(self, x):
        batch_size, seq_len = x.size()
        
        # Embedding and LSTM forward pass
        embedded = self.item_embeddings(x)  # (batch_size, seq_len, embedding_dim)
        lstm_out, _ = self.lstm(embedded)  # (batch_size, seq_len, hidden_size)
        
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
        sequences, targets = batch
        logits = self(sequences)
        loss = self.criterion(logits, targets.float())
        self._compute_metrics(logits, targets, self.train_metrics, 'train_')
        self.log('train_loss', loss)
        return loss
    
    def validation_step(self, batch, batch_idx):
        sequences, targets = batch
        logits = self(sequences)
        loss = self.criterion(logits, targets.float())
        self._compute_metrics(logits, targets, self.val_metrics, 'val_')
        self.log('val_loss', loss)
        return loss
    
    def test_step(self, batch, batch_idx):
        sequences, targets = batch
        logits = self(sequences)
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
        return DataLoader(self.train_dataset, batch_size=self.batch_size, shuffle=True)
    
    def val_dataloader(self):
        return DataLoader(self.val_dataset, batch_size=self.batch_size)
    
    def test_dataloader(self):
        return DataLoader(self.test_dataset, batch_size=self.batch_size)

# Example usage
if __name__ == "__main__":

    # Define parameters
    num_sequences = 100  # Number of sequences
    sequence_length = 20  # Length of each sequence
    sequence_start = 1  # Start range
    sequence_end = 49  # End range

    # Generate the sequences
    import json
    pathlist=['data/all_checkinsdict.json','data/gowalladict.json']
    path=pathlist[0]

    def extract_dict():
        # Read dictionary from the JSON file
        with open(path, 'r') as file:
            loaded_dict = json.load(file)
        itemlist=[]
        for i in loaded_dict:
            itemlist.append(loaded_dict[i])
        return itemlist
    
    def findUniqueItem():
        uniqueItem=[]
        with open(path, 'r') as file:
            loaded_dict = json.load(file)
        for i in loaded_dict:
            for j in loaded_dict[i]:
                if j not in uniqueItem:
                    uniqueItem.append(j)
            #     print(j)
            # print(i)
            # count+=1
            # if count>2:
            #     break
        return len(uniqueItem)

    user_sequences=extract_dict()
    NUM_ITEMS=findUniqueItem() + 1

    # Hyperparameters
    SEQUENCE_LENGTH = 40
    FUTURE_WINDOW = 3
    EMBEDDING_DIM = 32
    HIDDEN_SIZE = 64
    NUM_LAYERS = 1
    BATCH_SIZE = 32
    MAX_EPOCHS = 10
    TOP_K = [5, 10, 20]

    # Create data module with specific split ratios
    data_module = RecommendationDataModule(
        user_sequences=user_sequences,
        num_items=NUM_ITEMS,
        sequence_length=SEQUENCE_LENGTH,
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