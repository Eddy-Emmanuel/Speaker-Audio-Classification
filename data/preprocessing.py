"""Audio loading + feature extraction: waveform -> normalized log-mel spectrogram.

This mirrors the original notebook's `load_and_prep_waveform`, with one change:
`max_len_samples` is now an explicit argument instead of a module-level global,
so the function has no hidden state and can be reused/tested independently.
"""

import torch
import torchaudio

from .visualization import plot_spectogram, plot_mel_spectrogram, plot_waveform, play_audio


def build_augmentation_pipeline() -> torch.nn.Sequential:
    """Time-stretch + frequency/time masking, applied to the mel spectrogram."""
    return torch.nn.Sequential(
        torchaudio.transforms.TimeStretch(0.8, fixed_rate=True),
        torchaudio.transforms.FrequencyMasking(freq_mask_param=80),
        torchaudio.transforms.TimeMasking(time_mask_param=80),
    )


def load_and_prep_waveform(
    path,
    config,
    max_len_samples: int,
    transform=None,
    plot_steps: bool = False,
    play_audio_steps: bool = False,
):
    """Load a .wav file and turn it into a fixed-length, normalized log-mel spectrogram.

    Pipeline: load -> resample to config.sample_rate -> mono mix-down ->
    pad/trim to `max_len_samples` -> mel spectrogram -> amplitude-to-dB ->
    optional augmentation -> per-sample standardization.

    Returns a tensor of shape (1, n_mels, time_frames).
    """
    audio_wave, sample_rate = torchaudio.load(path)
    # Shape: (channels, num_samples), e.g. (1, 44100) for mono or (2, 44100) for stereo

    if plot_steps:
        plot_waveform(audio_wave[0:1], sample_rate,
                      title=f"Raw Load  |  sr={sample_rate} Hz  |  shape={tuple(audio_wave.shape)}")
    if play_audio_steps:
        play_audio(audio_wave, sample_rate, label="Before processing")

    if config.sample_rate != sample_rate:
        resampler = torchaudio.transforms.Resample(orig_freq=sample_rate, new_freq=config.sample_rate)
        audio_wave = resampler(audio_wave)
        # Shape: (channels, num_resampled_samples), e.g. (1, 16000) if resampled to 16kHz
        if plot_steps:
            plot_waveform(audio_wave, config.sample_rate,
                          title=f"Resampled  |  {sample_rate} → {config.sample_rate} Hz  |  shape={tuple(audio_wave.shape)}")
        if play_audio_steps:
            play_audio(audio_wave, config.sample_rate, label=f"After resample ({sample_rate} → {config.sample_rate} Hz)")

    if audio_wave.shape[0] > 1:
        audio_wave = audio_wave.mean(dim=0, keepdim=True)
        # Shape: (1, num_samples) — stereo/multi-channel collapsed to mono
        if plot_steps:
            plot_waveform(audio_wave, config.sample_rate,
                          title=f"Mono Mix-down  |  shape={tuple(audio_wave.shape)}")
        if play_audio_steps:
            play_audio(audio_wave, config.sample_rate, label="After mono mix-down")

    original_len = audio_wave.shape[-1]
    if audio_wave.shape[-1] < max_len_samples:
        pad_width = max_len_samples - audio_wave.shape[-1]
        audio_wave = torch.nn.functional.pad(audio_wave, pad=(0, pad_width), mode="constant", value=0)
        action = f"Padded +{pad_width} samples"
    else:
        audio_wave = audio_wave[:, :max_len_samples]
        action = f"Trimmed from {original_len} → {max_len_samples} samples"
    # Shape: (1, max_len_samples) — fixed-length waveform regardless of original duration

    if plot_steps:
        plot_waveform(audio_wave, config.sample_rate,
                      title=f"{action}  |  shape={tuple(audio_wave.shape)}")
    if play_audio_steps:
        play_audio(audio_wave, config.sample_rate, label=f"After {action.lower()}")

    mel_transform = torchaudio.transforms.MelSpectrogram(
        sample_rate=config.sample_rate,
        n_fft=config.n_fft,
        hop_length=config.n_fft // 2,
        n_mels=config.n_mels,
    )
    mel = mel_transform(audio_wave)
    # Shape: (1, n_mels, time_frames), e.g. (1, 64, 313)

    if plot_steps:
        plot_spectogram(mel)

    amplitude_to_db = torchaudio.transforms.AmplitudeToDB(top_db=80)
    audio_mel_in_db = amplitude_to_db(mel)
    # Shape: (1, n_mels, time_frames) — unchanged, values now in dB scale

    if transform is not None:
        audio_mel_in_db = transform(audio_mel_in_db)
        # Shape: (1, n_mels, time_frames) — unchanged unless transform modifies spatial dims

    audio_mel_in_db_norm = (audio_mel_in_db - audio_mel_in_db.mean()) / (audio_mel_in_db.std() + 1e-6)
    # Shape: (1, n_mels, time_frames) — unchanged, values zero-mean unit-variance normalized

    if plot_steps:
        plot_mel_spectrogram(audio_mel_in_db_norm, config.sample_rate, config.n_fft // 2,
                             title=f"Mel Spectrogram (dB, top_db=80)  |  shape={tuple(audio_mel_in_db.shape)}")

    return audio_mel_in_db_norm
    # Returns: (1, n_mels, time_frames)
