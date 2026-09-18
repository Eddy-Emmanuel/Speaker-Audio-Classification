# Speaker Audio Classification

Speaker identification from raw audio: a two-speaker classification task built on log-mel spectrograms, comparing three model architectures head-to-head — a CNN, a from-scratch Transformer with rotary positional embeddings (RoPE), and a stacked LSTM+GRU.

## How it works

1. **Data.** Audio clips are scanned from a directory of `<speaker_label>/*.wav` files and filtered down to two speakers (`Speaker_0001`, `Speaker_0002`) for binary classification.
2. **Preprocessing.** Each clip is resampled to 16kHz, mixed down to mono, padded/trimmed to a fixed length, and converted into a log-mel spectrogram (64 mel bins), then standardized (zero mean, unit variance). Optional augmentation applies time-stretch and frequency/time masking.
3. **Models.** Three independent architectures are trained on the same spectrograms and compared:
   - **CNN** — a small 2D convolutional encoder over the spectrogram "image."
   - **Transformer** — a from-scratch multi-head self-attention encoder using Rotary Positional Embeddings (RoPE), treating each spectrogram as a sequence of mel frames.
   - **LSTM + GRU** — a stacked recurrent model over the same frame sequence.
4. **Training.** Each model is trained with `CrossEntropyLoss`, AdamW, and a cosine LR schedule with warmup, tracked with Accuracy / F1 / Precision / AUROC (via `torchmetrics`), and stopped early on validation loss plateau.

## Repository Structure

```
speaker-audio-classification/
├── configs/
│   └── config.py            # AudioConfig — all hyperparameters & paths
├── data/
│   ├── manifest.py           # build (path, label) dataframe from disk, find max clip length
│   ├── preprocessing.py      # waveform -> normalized log-mel spectrogram
│   ├── dataset.py            # AudioDataset (PyTorch Dataset)
│   └── visualization.py      # waveform/spectrogram plotting + audio playback helpers
├── models/
│   ├── cnn.py                 # ModelCNN
│   ├── transformer.py         # RoPE, ModelAttention, AudioTransformerBlock, AudioTransformer
│   └── rnn.py                 # LstmModel
├── training/
│   ├── trainer.py             # Trainer: train/eval loop + early stopping + checkpointing
│   └── metrics.py             # torchmetrics factory (accuracy, f1, precision, auroc)
├── utils.py                   # clear_memory() — GPU/CPU cleanup between model runs
├── train.py                   # entry point: builds data, trains all three models
├── requirements.txt
└── speaker-audio-classification.ipynb   # original exploratory notebook
```

This mirrors the original notebook's logic, reorganized into importable modules. The one behavioral change: `load_and_prep_waveform` now takes `max_len_samples` as an explicit argument instead of relying on a module-level global, so it can be reused/tested independently of a specific run.

## Setup

```bash
pip install -r requirements.txt
```

Update `AudioConfig.audio_base_dir` in `configs/config.py` to point at your local copy of the [Speaker Recognition Audio Dataset](https://www.kaggle.com/datasets/vjcalling/speaker-recognition-audio-dataset) (expects `<audio_base_dir>/<speaker_label>/*.wav`).

## Usage

Train and compare all three models end-to-end:

```bash
python train.py
```

This will, in order:
1. Build the manifest and train/val split (90/10, stratified).
2. Train the LSTM+GRU model, then the Transformer, then the CNN — each with early stopping, saving its best checkpoint as `<model_name>.pt`.
3. Clear GPU memory between each model so all three can run back-to-back on the same machine.

To use a single model, architecture, or preprocessing step in your own script:

```python
from configs import AudioConfig
from data import create_audio_dataframe, load_and_prep_waveform
from models import AudioTransformer

df = create_audio_dataframe(AudioConfig.audio_base_dir)
mel = load_and_prep_waveform(path=df["path"][0], config=AudioConfig, max_len_samples=64_000)
model = AudioTransformer(AudioConfig)
```

## Configuration

All hyperparameters live in `configs/config.py` (`AudioConfig`): mel bins, FFT size, sample rate, model dimensions, batch size, number of epochs, early-stopping patience, and the two CUDA devices models are trained on (`device_0` for CNN/Transformer, `device_1` for the LSTM/GRU — trained in parallel on a multi-GPU setup, or update to a single device if you don't have two GPUs).

## Notes

- The CNN, Transformer, and LSTM/GRU each expect the input in a different shape (`_get_output` in `Trainer` handles this), so the `Trainer` class is architecture-agnostic and shared across all three runs.
- `AudioConfig.device_1` is only used by the LSTM/GRU model; if you're on a single-GPU or CPU-only machine, set both `device_0` and `device_1` to the same device.
- Checkpoints (`cnn_model.pt`, `audio_transformer.pt`, `lstm_model.pt`) are written to the working directory by the `Trainer`'s early-stopping logic.
