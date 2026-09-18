"""Generic train/eval loop with early stopping, used for all three model
architectures (CNN, Transformer, LSTM+GRU).
"""

import torch
from tqdm import tqdm


class Trainer:
    def __init__(self, model, train_dl, val_dl, config, optimizer, criterion, lr_scheduler, metrics):
        print("Initializing trainer")
        self.model = model
        self.config = config
        self.train_dl = train_dl
        self.val_dl = val_dl
        self.metrics = metrics
        self.optimizer = optimizer
        self.criterion = criterion
        self.lr_scheduler = lr_scheduler
        self.device = self.config.device_1 if self.model.name == "lstm_model" else self.config.device_0

        self.best_val_loss = float("inf")
        self.patience_counter = 0
        self.checkpoint_path = self.model.name + ".pt"

    def _compute_metrics(self, output, y):
        probs = output.softmax(dim=1)
        preds = output.argmax(dim=1)
        return {
            "accuracy": self.metrics["accuracy"].to(self.device)(preds, y).item(),
            "f1_score": self.metrics["f1_score"].to(self.device)(preds, y).item(),
            "precision": self.metrics["precision"].to(self.device)(preds, y).item(),
            "auroc": self.metrics["auroc"].to(self.device)(probs, y).item(),
        }

    def _get_output(self, X):
        # The CNN expects a (B, 1, n_mels, T) "image"; the sequence models
        # (Transformer/LSTM) expect (B, T, n_mels).
        return self.model(X) if self.model.name == "cnn_model" else self.model(X.squeeze(1))

    def train(self):
        total_loss = 0
        total_metrics = {"accuracy": 0.0, "f1_score": 0.0, "precision": 0.0, "auroc": 0.0}

        self.model.train()
        for batch_X, batch_y in self.train_dl:
            X, y = batch_X.to(self.device), batch_y.to(self.device)
            self.optimizer.zero_grad()
            output = self._get_output(X)

            loss = self.criterion(output, y)

            loss.backward()
            self.optimizer.step()
            self.lr_scheduler.step()

            total_loss += loss.item()
            batch_metrics = self._compute_metrics(output, y)

            for k in total_metrics:
                total_metrics[k] += batch_metrics[k]

        n = len(self.train_dl)
        avg_metrics = {k: v / n for k, v in total_metrics.items()}

        return total_loss / n, avg_metrics

    def eval(self):
        total_loss = 0
        total_metrics = {"accuracy": 0.0, "f1_score": 0.0, "precision": 0.0, "auroc": 0.0}

        self.model.eval()
        with torch.no_grad():
            for batch_X, batch_y in self.val_dl:
                X, y = batch_X.to(self.device), batch_y.to(self.device)
                output = self._get_output(X)

                loss = self.criterion(output, y)
                total_loss += loss.item()

                batch_metrics = self._compute_metrics(output, y)
                for k in total_metrics:
                    total_metrics[k] += batch_metrics[k]

        n = len(self.val_dl)
        avg_metrics = {k: v / n for k, v in total_metrics.items()}

        return total_loss / n, avg_metrics

    def _early_stopping(self, val_loss):
        if val_loss < self.best_val_loss:
            self.best_val_loss = val_loss
            self.patience_counter = 0
            torch.save(self.model.state_dict(), self.checkpoint_path)
            return False
        else:
            self.patience_counter += 1
            if self.patience_counter >= self.config.patience:
                return True
        return False

    def fit(self):
        epoch_pbar = tqdm(range(1, self.config.num_epochs + 1), desc="Training", leave=True)

        for epoch in epoch_pbar:
            train_loss, train_metrics = self.train()
            val_loss, val_metrics = self.eval()

            epoch_pbar.set_postfix({
                "train_loss": f"{train_loss:.4f}",
                "train_acc": f"{train_metrics['accuracy']:.4f}",
                "train_f1": f"{train_metrics['f1_score']:.4f}",
                "train_prec": f"{train_metrics['precision']:.4f}",
                "train_auc": f"{train_metrics['auroc']:.4f}",

                "val_loss": f"{val_loss:.4f}",
                "val_acc": f"{val_metrics['accuracy']:.4f}",
                "val_f1": f"{val_metrics['f1_score']:.4f}",
                "val_prec": f"{val_metrics['precision']:.4f}",
                "val_auc": f"{val_metrics['auroc']:.4f}",
            })

            if self._early_stopping(val_loss):
                tqdm.write(f"\n⏹ Early stopping at epoch {epoch} — val loss didn't improve for {self.config.patience} epochs")
                tqdm.write(f"✅ Best val loss: {self.best_val_loss:.4f} — model saved to {self.checkpoint_path}")
                break
