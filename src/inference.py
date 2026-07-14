import torch
import numpy as np
from PIL import Image
from torchvision import transforms as tr

from .config import cfg
from .vocab import Vocab
from .model import ImageCaptioningModel

_transform = tr.Compose([
    tr.Resize(256),
    tr.CenterCrop(cfg.img_size),
    tr.ToTensor(),
    tr.Normalize(mean=cfg.channel_mean, std=cfg.channel_std),
])


def _remap_notebook_keys(state: dict) -> dict:
    if any(k.startswith("encoder.") for k in state):
        return state

    new_state = {}
    for k, v in state.items():
        if k.startswith("img_fe."):
            k = "encoder." + k[len("img_fe."):]
        elif k.startswith("text_fe."):
            k = "decoder." + k[len("text_fe."):]
        elif k.startswith("classifier.fc."):
            k = "head.1." + k[len("classifier.fc."):]
        new_state[k] = v
    return new_state


def load_artifacts(device="cpu"):
    vocab = Vocab.load(cfg.vocab_path)
    model = ImageCaptioningModel(
        vocab_size=len(vocab),
        hidden_size=cfg.hidden_size,
        num_layers=cfg.num_layers,
        dropout=cfg.dropout,
        pad_id=vocab.pad_id,
    )
    ckpt = torch.load(cfg.checkpoint_path, map_location=device)
    state = ckpt.get("model_state_dict", ckpt)
    state = _remap_notebook_keys(state)
    model.load_state_dict(state)
    model.to(device).eval()
    return model, vocab


def _sample(probs, top_k=None, top_p=None, temperature=1.0):
    if temperature != 1.0:
        probs = probs ** (1.0 / temperature)
        probs = probs / probs.sum()
    if top_k is not None:
        p, idx = torch.topk(probs, top_k)
        return idx[torch.multinomial(p / p.sum(), 1)].item()
    if top_p is not None:
        p, idx = torch.sort(probs, descending=True)
        keep = max((torch.cumsum(p, 0) <= top_p).sum().item(), 1)
        return idx[torch.multinomial(p[:keep] / p[:keep].sum(), 1)].item()
    return torch.multinomial(probs, 1).item()


@torch.no_grad()
def generate_caption(model, vocab, image: Image.Image, device="cpu",
                     top_k=None, top_p=None, temperature=1.0,
                     max_len=cfg.max_seq_len) -> str:
    img = _transform(image.convert("RGB"))[None].to(device)
    feats = model.encoder(img)
    state = model.decoder.init_state(feats)

    token = torch.tensor([vocab.bos_id], device=device)
    result = []
    for _ in range(max_len):
        hidden, state = model.decoder.step(token, state)
        probs = torch.exp(model.head(hidden)[0])
        next_id = _sample(probs, top_k=top_k, top_p=top_p, temperature=temperature)
        if next_id == vocab.eos_id:
            break
        result.append(next_id)
        token = torch.tensor([next_id], device=device)

    return vocab.decode(result)
