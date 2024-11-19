import pytorch_lightning as pl
import torch
import torch.nn as nn
from torchmetrics import Metric
from typing import List, Set

class PrecisionRecallAtK(Metric):
    def __init__(self, k: int):
        super().__init__()
        self.k = k
        self.add_state("precision_sum", default=torch.tensor(0.0), dist_reduce_fx="sum")
        self.add_state("recall_sum", default=torch.tensor(0.0), dist_reduce_fx="sum")
        self.add_state("total", default=torch.tensor(0), dist_reduce_fx="sum")
    
    def update(self, preds: torch.Tensor, target: torch.Tensor):
        # preds: (batch_size, num_items) - logits
        # target: (batch_size, num_items) - binary vector
        
        # Get top-k predictions
        _, top_k_indices = torch.topk(preds, self.k, dim=1)
        
        for i in range(len(preds)):
            pred_set = set(top_k_indices[i].tolist())
            true_set = set(torch.where(target[i] > 0)[0].tolist())
            
            if len(true_set) > 0:
                intersection = len(pred_set.intersection(true_set))
                precision = intersection / self.k
                recall = intersection / len(true_set)
                
                self.precision_sum += precision
                self.recall_sum += recall
                self.total += 1
    
    def compute(self):
        if self.total == 0:
            return torch.tensor(0.0), torch.tensor(0.0)
        return (self.precision_sum / self.total, self.recall_sum / self.total)

class NextFutureRecommender(pl.LightningModule):
    def __init__(
        self, 
        num_items: int,
        embedding_dim: int,
        hidden_size: int,
        num_layers: int,
        dropout: float = 0.1,
        top_k: List[int] = [5, 10, 20]
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
        
        # Final prediction layer
        self.prediction = nn.Sequential(
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size, num_items),
        )
        
        # Binary Cross Entropy loss with logits
        self.criterion = nn.BCEWithLogitsLoss()
        
        # Metrics for different k values
        self.train_metrics = nn.ModuleDict({
            f'precision_recall@{k}': PrecisionRecallAtK(k) for k in top_k
        })
        self.val_metrics = nn.ModuleDict({
            f'precision_recall@{k}': PrecisionRecallAtK(k) for k in top_k
        })
        
    def forward(self, x):
        # x shape: (batch_size, sequence_length)
        embedded = self.item_embeddings(x)  # (batch_size, sequence_length, embedding_dim)
        
        # LSTM forward pass
        lstm_out, _ = self.lstm(embedded)  # (batch_size, sequence_length, hidden_size)
        
        # Use only the last output
        last_hidden = lstm_out[:, -1, :]  # (batch_size, hidden_size)
        
        # Final prediction
        logits = self.prediction(last_hidden)  # (batch_size, num_items)
        return logits
    
    def _compute_metrics(self, logits, targets, metrics_dict):
        for name, metric in metrics_dict.items():
            precision, recall = metric(logits, targets)
            self.log(f'{name}_precision', precision)
            self.log(f'{name}_recall', recall)
    
    def training_step(self, batch, batch_idx):
        sequences, targets = batch
        logits = self(sequences)
        loss = self.criterion(logits, targets.float())
        
        # Compute metrics
        self._compute_metrics(logits, targets, self.train_metrics)
        
        self.log('train_loss', loss)
        return loss
    
    def validation_step(self, batch, batch_idx):
        sequences, targets = batch
        logits = self(sequences)
        loss = self.criterion(logits, targets.float())
        
        # Compute metrics
        self._compute_metrics(logits, targets, self.val_metrics)
        
        self.log('val_loss', loss)
        return loss
    
    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=0.001)
