import re
import json
from collections import Counter

SPECIALS = {"<UNK>": 0, "<BOS>": 1, "<EOS>": 2, "<PAD>": 3}


def tokenize(text: str) -> list[str]:
    text = text.lower()
    text = re.sub(r"[^\w\s]|_", " ", text)
    text = text.strip()
    tokens = re.split(r"\s+", text) if text else []
    return ["<BOS>"] + tokens + ["<EOS>"]


class Vocab:
    def __init__(self, tok_to_ind: dict[str, int]):
        self.tok_to_ind = tok_to_ind
        self.ind_to_tok = {v: k for k, v in tok_to_ind.items()}

    def __len__(self):
        return len(self.tok_to_ind)

    @property
    def pad_id(self): return self.tok_to_ind["<PAD>"]
    @property
    def bos_id(self): return self.tok_to_ind["<BOS>"]
    @property
    def eos_id(self): return self.tok_to_ind["<EOS>"]
    @property
    def unk_id(self): return self.tok_to_ind["<UNK>"]

    def encode(self, text: str) -> list[int]:
        return [self.tok_to_ind.get(t, self.unk_id) for t in tokenize(text)]

    def decode(self, ids: list[int]) -> str:
        specials = set(SPECIALS.values())
        return " ".join(self.ind_to_tok[i] for i in ids if i not in specials)

    # --- сериализация ---
    def save(self, path):
        with open(path, "w") as f:
            json.dump(self.tok_to_ind, f, ensure_ascii=False)

    @classmethod
    def load(cls, path):
        with open(path) as f:
            return cls(json.load(f))

    @classmethod
    def build(cls, captions: list[str], min_freq: int = 3):
        freq = Counter()
        for cap in captions:
            freq.update(tokenize(cap)[1:-1])
        tok_to_ind = dict(SPECIALS)
        for tok, f in freq.items():
            if f >= min_freq:
                tok_to_ind[tok] = len(tok_to_ind)
        return cls(tok_to_ind)
