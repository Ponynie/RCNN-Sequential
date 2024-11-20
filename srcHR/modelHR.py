import pytorch_lightning as pl
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
from torchmetrics.retrieval import RetrievalHitRate
from torcheval.metrics.functional import hit_rate

import torch

def ndcg(predictions: torch.Tensor, 
         target: torch.Tensor, 
         k: int = 10) -> torch.Tensor:
    """
    Compute Normalized Discounted Cumulative Gain (NDCG) for each sample in batch.
    
    Args:
        predictions: Model predictions/logits tensor of shape (batch_size, num_items)
        target: Ground truth labels tensor of shape (batch_size,)
        k: Number of items to consider for NDCG calculation
    
    Returns:
        Tensor containing NDCG@k scores for each sample in batch
    """
    batch_size = predictions.size(0)
    
    # Get top k predicted items
    _, indices = torch.topk(predictions, k, dim=1)
    
    # Create a binary tensor indicating if prediction matches target
    hits = (indices == target.unsqueeze(1)).float()
    
    # Calculate position-based discounts
    position_discounts = torch.log2(torch.arange(k, device=predictions.device).float() + 2.0)
    dcg = (hits / position_discounts).sum(dim=1)
    
    # Calculate ideal DCG (target in first position)
    idcg = (1 / torch.log2(torch.tensor(2.0, device=predictions.device))).expand(batch_size)
    
    # Return NDCG scores
    return dcg / idcg

class RCNN_NextItem(pl.LightningModule):
    def __init__(self, 
                 num_items: int,
                 embedding_dim: int,
                 hidden_size: int,
                 num_layers: int,
                 lstm_dropout: float = 0.1,
                 conv_out_channels: int = 8,
                 horizontal_filter_size: int = 64,
                 vertical_filter_size: int = 4,
                 learning_rate: float = 0.001,
                 weight_decay: float = 0.001):
         
        super().__init__()
        self.save_hyperparameters()
        
        self.learning_rate = learning_rate
        self.weight_decay = weight_decay
        self.num_items = num_items
        
        # Item embedding layer
        self.item_embeddings = nn.Embedding(num_items, embedding_dim)
        
        # LSTM layer
        self.lstm = nn.LSTM(
            input_size=embedding_dim,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=lstm_dropout if num_layers > 1 else 0
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
        self.prediction = nn.Linear(2 * hidden_size + conv_out_channels, num_items)
        
        # Cross Entropy loss (target should be class indices)
        self.criterion = nn.CrossEntropyLoss()
        
    def forward(self, x, lengths):
        batch_size, seq_len = x.size()
        
        # Embedding and LSTM forward pass
        embedded = self.item_embeddings(x)
        packed_embedded = pack_padded_sequence(embedded, lengths, batch_first=True, enforce_sorted=False)
        packed_lstm_out, _ = self.lstm(packed_embedded)
        lstm_out, _ = pad_packed_sequence(packed_lstm_out, batch_first=True)
        
        # Horizontal Convolution
        horizontal_input = lstm_out.unsqueeze(1)
        horizontal_conv_out = F.relu(self.horizontal_conv(horizontal_input))[:, :, -1, :]
        horizontal_conv_out = torch.sum(horizontal_conv_out, dim=2)
        
        # Vertical Convolution
        vertical_input = lstm_out.unsqueeze(1)[:, :, -self.vertical_filter_size:, :]
        vertical_conv_out = F.relu(self.vertical_conv(vertical_input))
        vertical_conv_out = vertical_conv_out.view(batch_size, -1)
        
        # Concatenate LSTM output with convolution outputs
        last_hidden = lstm_out[:, -1, :]
        vertical_conv_out = torch.mul(vertical_conv_out, last_hidden)
        
        combined = torch.cat([horizontal_conv_out, last_hidden, vertical_conv_out], dim=1)
        
        # Final prediction (logits)
        logits = self.prediction(combined)
        return logits
    
    def training_step(self, batch, batch_idx):
        sequences, target, length = batch  # target is class index
        logits = self(sequences, length)
        loss = self.criterion(logits, target)
        
        # Compute HR@10
        hr = hit_rate(logits, target, k=10).mean()
        ndcg_sc = ndcg(logits, target, k=10).mean()
        # Log metrics
        self.log('train_loss', loss, on_step=True, on_epoch=True, logger=True, batch_size=sequences.size(0))
        self.log('train_hr@10', hr, on_epoch=True, logger=True, batch_size=sequences.size(0))
        self.log('train_ndcg@10', ndcg_sc, on_epoch=True, logger=True, batch_size=sequences.size(0))
        return loss
    
    def validation_step(self, batch, batch_idx):
        sequences, target, length = batch  # target is class index
        logits = self(sequences, length)
        loss = self.criterion(logits, target)
        
        # Compute HR@10
        hr = hit_rate(logits, target, k=10).mean()
        # Compute NDCG@10
        ndcg_sc = ndcg(logits, target, k=10).mean()
        
        # Log metrics
        self.log('val_loss', loss, on_epoch=True, logger=True, prog_bar=True, batch_size=sequences.size(0))
        self.log('val_hr@10', hr, on_epoch=True, logger=True, batch_size=sequences.size(0))
        self.log('val_ndcg@10', ndcg_sc, on_epoch=True, logger=True, batch_size=sequences.size(0))
        return loss
    
    def test_step(self, batch, batch_idx):
        sequences, target, length = batch  # target is class index
        logits = self(sequences, length)
        loss = self.criterion(logits, target)
        
        # Compute HR@10
        hr = hit_rate(logits, target, k=10).mean()
        # Compute NDCG@10
        ndcg_sc = ndcg(logits, target, k=10).mean()
        
        # Log metrics
        self.log('test_loss', loss, on_epoch=True, logger=True, prog_bar=True, batch_size=sequences.size(0))
        self.log('test_hr@10', hr, on_epoch=True, logger=True, batch_size=sequences.size(0))
        self.log('test_ndcg@10', ndcg_sc, on_epoch=True, logger=True, batch_size=sequences.size(0))
        return loss
    
    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=self.learning_rate, weight_decay=self.weight_decay)