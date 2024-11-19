import pytorch_lightning as pl
import torch
import torch.nn as nn
from torchmetrics.retrieval import RetrievalPrecision, RetrievalRecall
from typing import List
import torch.nn.functional as F 
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
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