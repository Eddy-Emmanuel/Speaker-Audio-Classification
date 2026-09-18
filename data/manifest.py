"""Build a dataframe manifest (path, label) from a directory of speaker audio."""

from glob import glob

import pandas as pd
import soundfile as sf


def create_audio_dataframe(base_dir: str) -> pd.DataFrame:
    """Recursively scan ``base_dir`` for .wav files.

    Expects a layout of ``base_dir/<speaker_label>/*.wav``, so the label is
    taken from the immediate parent directory of each file.
    """
    rows = []
    for path in glob(base_dir + "/**/*.wav", recursive=True):
        label = path.split("/")[-2]
        rows.append([path, label])
    return pd.DataFrame(rows, columns=["path", "label"])


def get_max_audio_duration_seconds(df: pd.DataFrame) -> float:
    """Return the longest clip duration (in seconds) across all paths in ``df``."""
    max_len = 0.0
    for path in df["path"]:
        f = sf.SoundFile(path)
        duration = len(f) / f.samplerate
        if duration > max_len:
            max_len = duration
    return max_len
