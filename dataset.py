"""PyTorch Dataset wrapping the (path, label) manifest."""

import torch
from torch.utils.data import Dataset


class AudioDataset(Dataset):
    """Loads a wav file on-the-fly and returns (mel_spectrogram, label).

    Args:
        df: dataframe with a "path" column and a "label_enc" (int) column.
        config: AudioConfig-like object (needs sample_rate, n_fft, n_mels).
        img_prep_func: callable(path, config, max_len_samples, transform=...) ->
            tensor, e.g. `data.preprocessing.load_and_prep_waveform`.
        max_len_samples: fixed waveform length (in samples) all clips are
            padded/trimmed to before feature extraction.
        transform: optional augmentation applied to the mel spectrogram.
    """

    def __init__(self, df, config, img_prep_func, max_len_samples: int, transform=None):
        self.X = df["path"]
        self.y = df["label_enc"]
        self.config = config
        self.transform = transform
        self.img_prep_func = img_prep_func
        self.max_len_samples = max_len_samples

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        X = self.img_prep_func(
            path=self.X[idx],
            config=self.config,
            max_len_samples=self.max_len_samples,
            transform=self.transform,
        )
        y = torch.tensor(self.y[idx], dtype=torch.long)
        return X.transpose(-2, -1), y
