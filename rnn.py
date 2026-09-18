"""Recurrent classifier: a stacked LSTM followed by a GRU over mel frames."""

import torch.nn as nn


class LstmModel(nn.Module):
    def __init__(self, config, name: str = "lstm_model"):
        super().__init__()
        self._name = name
        self.input_proj = nn.Linear(in_features=config.n_mels, out_features=config.embed_dim)
        self.block = nn.ModuleList([
            nn.LSTM(input_size=config.embed_dim, hidden_size=config.embed_dim,
                    num_layers=config.lstm_layer, batch_first=True),
            nn.GRU(input_size=config.embed_dim, hidden_size=config.embed_dim,
                   num_layers=config.gru_layer, batch_first=True),
        ])
        self.output_proj = nn.Linear(config.embed_dim, config.num_class)

    @property
    def name(self):
        return self._name

    def forward(self, x):
        x = self.input_proj(x)

        for block in self.block:
            x, _ = block(x)

        return self.output_proj(x[:, -1, :])
