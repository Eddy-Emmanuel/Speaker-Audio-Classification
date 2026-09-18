"""Plotting and playback helpers used while inspecting audio preprocessing steps."""

import matplotlib.pyplot as plt
import torch
from IPython.display import Audio, display


def plot_waveform(waveform, sample_rate, title="Waveform"):
    plt.figure(figsize=(12, 4))
    time_axis = torch.arange(0, waveform.shape[-1]) / sample_rate
    plt.plot(time_axis.numpy(), waveform.squeeze().numpy())
    plt.title(title)
    plt.xlabel("Time (s)")
    plt.ylabel("Amplitude")
    plt.tight_layout()
    plt.show()


def plot_spectogram(spec):
    plt.figure(figsize=(12, 4))
    plt.imshow(spec.squeeze().numpy(), aspect="auto", origin="lower")
    plt.axis("off")
    plt.title(f"Mel Spectrogram (linear)  |  shape={tuple(spec.shape)}")
    plt.xlabel("Time Frames")
    plt.ylabel("Mel Bins")
    plt.tight_layout()
    plt.show()


def plot_mel_spectrogram(mel_db, sample_rate, hop_length, title="Mel Spectrogram (dB)"):
    plt.figure(figsize=(12, 4))
    plt.imshow(mel_db.squeeze().numpy(), aspect="auto", origin="lower")
    plt.axis("off")
    plt.title(title)
    plt.xlabel("Time (s)")
    plt.ylabel("Mel Filter Banks")
    plt.tight_layout()
    plt.show()


def play_audio(waveform, sample_rate, label="Audio"):
    print(f"▶ {label}")
    display(Audio(waveform.squeeze().numpy(), rate=sample_rate))
