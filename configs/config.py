"""Central configuration for the speaker audio classification pipeline."""


class AudioConfig:
    # --- Audio / feature extraction ---
    n_mels = 64
    n_fft = 512
    sample_rate = 16_000

    # --- Model architecture ---
    embed_dim = 1024
    max_seq_len = 2048
    num_head = 16
    img_size = 224
    patch_size = 16
    num_transformer_layers = 2
    lstm_layer = 1
    gru_layer = 1

    # --- Training ---
    batch_size = 2
    num_class = 2
    num_epochs = 20
    patience = 5

    # --- Devices ---
    device_0 = "cuda:0"
    device_1 = "cuda:1"

    # --- Data ---
    audio_base_dir = "/kaggle/input/datasets/vjcalling/speaker-recognition-audio-dataset/50_speakers_audio_data"
    speakers_to_use = ["Speaker_0002", "Speaker_0001"]
