from dataclasses import dataclass
from pathlib import Path

@dataclass
class Config:
    # paths
    data_dir: Path = Path("./data")
    weights_dir: Path = Path("./weights")
    checkpoint_path: Path = Path("./weights/model.pt")
    vocab_path: Path = Path("./weights/vocab.json")
    glove_path: Path = Path("./glove.840B.300d.txt")

    # image
    img_size: int = 224
    channel_mean: tuple = (0.485, 0.456, 0.406)
    channel_std: tuple = (0.229, 0.224, 0.225)

    # vocab
    min_freq: int = 3

    # model
    embed_dim: int = 300
    hidden_size: int = 512
    num_layers: int = 2
    dropout: float = 0.3
    freeze_cnn: bool = True

    # training
    batch_size: int = 64
    lr: float = 3e-4
    weight_decay: float = 1e-5
    epochs: int = 30
    min_lr: float = 1e-6

    # generation
    max_seq_len: int = 25

cfg = Config()
