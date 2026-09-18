"""Entry point: build the dataset, then train and compare the CNN,
Transformer (RoPE), and LSTM+GRU speaker classifiers.

Usage:
    python train.py
"""

import os

import torch
import torch.nn as nn
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader
from transformers import get_cosine_schedule_with_warmup

from configs import AudioConfig
from data import AudioDataset, create_audio_dataframe, get_max_audio_duration_seconds, load_and_prep_waveform
from models import ModelCNN, AudioTransformer, LstmModel
from training import Trainer, build_metrics
from utils import clear_memory


def build_dataloaders(config):
    audio_df = create_audio_dataframe(config.audio_base_dir)
    filtered_df = audio_df[audio_df["label"].isin(config.speakers_to_use)].reset_index(drop=True)

    # NOTE: matches the original notebook, which computes the max clip length
    # from the *full* speaker set (audio_df), not just the two speakers used
    # for training. This just sets a (generously long) fixed padding length.
    max_len_samples = int(get_max_audio_duration_seconds(audio_df) * config.sample_rate)

    label_enc = LabelEncoder()
    filtered_df["label_enc"] = label_enc.fit_transform(filtered_df["label"])
    idx2label = dict(enumerate(label_enc.classes_))
    print("Label mapping:", idx2label)

    train_df, val_df = train_test_split(
        filtered_df, test_size=0.1, shuffle=True, stratify=filtered_df["label_enc"]
    )
    train_df.reset_index(drop=True, inplace=True)
    val_df.reset_index(drop=True, inplace=True)

    train_ds = AudioDataset(train_df, config, load_and_prep_waveform, max_len_samples)
    val_ds = AudioDataset(val_df, config, load_and_prep_waveform, max_len_samples)

    num_workers = os.cpu_count() or 0
    train_dl = DataLoader(train_ds, batch_size=config.batch_size, shuffle=True, num_workers=num_workers)
    val_dl = DataLoader(val_ds, batch_size=config.batch_size, shuffle=False, num_workers=num_workers)

    return train_dl, val_dl, idx2label


def build_optimizer_and_scheduler(model, config, num_training_steps):
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)
    scheduler = get_cosine_schedule_with_warmup(
        optimizer=optimizer,
        num_warmup_steps=int(0.1 * num_training_steps),
        num_training_steps=num_training_steps,
    )
    return optimizer, scheduler


def train_one_model(model, train_dl, val_dl, config, criterion, metrics):
    num_training_steps = config.num_epochs * len(train_dl)
    optimizer, scheduler = build_optimizer_and_scheduler(model, config, num_training_steps)

    trainer = Trainer(
        model=model,
        train_dl=train_dl,
        val_dl=val_dl,
        config=config,
        optimizer=optimizer,
        criterion=criterion,
        lr_scheduler=scheduler,
        metrics=metrics,
    )
    trainer.fit()
    clear_memory(model=model, optimizer=optimizer, trainer=trainer)


def main():
    config = AudioConfig
    train_dl, val_dl, idx2label = build_dataloaders(config)

    criterion = nn.CrossEntropyLoss()
    metrics = build_metrics(config.num_class)

    print("\n=== Training LSTM+GRU model ===")
    lstm_model = LstmModel(config).to(config.device_1)
    train_one_model(lstm_model, train_dl, val_dl, config, criterion, metrics)

    print("\n=== Training Transformer (RoPE) model ===")
    audio_transformer = AudioTransformer(config).to(config.device_0)
    train_one_model(audio_transformer, train_dl, val_dl, config, criterion, metrics)

    print("\n=== Training CNN model ===")
    cnn_model = ModelCNN().to(config.device_0)
    train_one_model(cnn_model, train_dl, val_dl, config, criterion, metrics)


if __name__ == "__main__":
    main()
