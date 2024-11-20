import pytorch_lightning as pl
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
from torchmetrics.retrieval import RetrievalHitRate

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
        
        # Final prediction layer - no sigmoid needed as we'll use CrossEntropyLoss
        self.prediction = nn.Linear(2 * hidden_size + conv_out_channels, num_items)
        
        # Cross Entropy loss (includes softmax)
        self.criterion = nn.CrossEntropyLoss()
        
        # Initialize HR@10 tracking variables
        self.train_hits = 0
        self.train_total = 0
        self.val_hits = 0
        self.val_total = 0
        self.test_hits = 0
        self.test_total = 0
        
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

    def compute_hr10(self, logits, targets):
        """Compute Hit Rate @10"""
        # Get top 10 predictions
        _, top10_indices = torch.topk(logits, k=10, dim=1)
        # Check if target is in top 10 predictions
        hits = sum([target in top10 for top10, target in zip(top10_indices, targets)])
        return hits, len(targets)
    
    def training_step(self, batch, batch_idx):
        sequences, target, length = batch
        logits = self(sequences, length)
        loss = self.criterion(logits, target)
        
        # Compute HR@10
        hits, total = self.compute_hr10(logits, target)
        self.train_hits += hits
        self.train_total += total
        
        # Log loss
        self.log('train_loss', loss, on_step=True, on_epoch=True, logger=True, batch_size=sequences.size(0))
        return loss
    
    def validation_step(self, batch, batch_idx):
        sequences, target, length = batch
        logits = self(sequences, length)
        loss = self.criterion(logits, target)
        
        # Compute HR@10
        hits, total = self.compute_hr10(logits, target)
        self.val_hits += hits
        self.val_total += total
        
        # Log loss
        self.log('val_loss', loss, prog_bar=True, logger=True, batch_size=sequences.size(0))
        return loss
    
    def test_step(self, batch, batch_idx):
        sequences, target, length = batch
        logits = self(sequences, length)
        loss = self.criterion(logits, target)
        
        # Compute HR@10
        hits, total = self.compute_hr10(logits, target)
        self.test_hits += hits
        self.test_total += total
        
        # Log loss
        self.log('test_loss', loss, prog_bar=True, logger=True, batch_size=sequences.size(0))
        return loss
    
    def on_train_epoch_end(self):
        # Compute and log HR@10 for training
        hr10 = self.train_hits / self.train_total if self.train_total > 0 else 0
        self.log('train_hr10', hr10, prog_bar=True)
        # Reset counters
        self.train_hits = 0
        self.train_total = 0
        
    def on_validation_epoch_end(self):
        # Compute and log HR@10 for validation
        hr10 = self.val_hits / self.val_total if self.val_total > 0 else 0
        self.log('val_hr10', hr10, prog_bar=True)
        # Reset counters
        self.val_hits = 0
        self.val_total = 0
        
    def on_test_epoch_end(self):
        # Compute and log HR@10 for test
        hr10 = self.test_hits / self.test_total if self.test_total > 0 else 0
        self.log('test_hr10', hr10, prog_bar=True)
        # Reset counters
        self.test_hits = 0
        self.test_total = 0
    
    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=self.learning_rate, weight_decay=self.weight_decay)