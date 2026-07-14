import torch
from torch import nn
from torchvision import models
from einops import rearrange


class ImageEncoder(nn.Module):
    def __init__(self, freeze: bool = True):
        super().__init__()
        backbone = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
        self.out_features = backbone.fc.in_features
        backbone.fc = nn.Identity()
        self.backbone = backbone
        if freeze:
            for p in self.backbone.parameters():
                p.requires_grad = False

    def forward(self, imgs):
        return self.backbone(imgs)


class TextDecoder(nn.Module):
    def __init__(self, vocab_size, img_dim, embed_dim=300,
                 hidden_size=512, num_layers=2, dropout=0.3, pad_id=3):
        super().__init__()
        self.num_layers = num_layers
        self.embed = nn.Embedding(vocab_size, embed_dim, padding_idx=pad_id)
        self.img_to_h0 = nn.Linear(img_dim, hidden_size)
        self.img_to_c0 = nn.Linear(img_dim, hidden_size)
        self.rnn = nn.LSTM(embed_dim, hidden_size, num_layers,
                           batch_first=True, dropout=dropout)

    def init_state(self, img_features, num_cap=1):
        h0 = torch.tanh(self.img_to_h0(img_features))
        c0 = torch.tanh(self.img_to_c0(img_features))
        h0 = h0[None, :, None, :].repeat(self.num_layers, 1, num_cap, 1)
        c0 = c0[None, :, None, :].repeat(self.num_layers, 1, num_cap, 1)
        h0 = rearrange(h0, "l bs cap h -> l (bs cap) h").contiguous()
        c0 = rearrange(c0, "l bs cap h -> l (bs cap) h").contiguous()
        return h0, c0

    def forward(self, texts, img_features):
        bs, num_cap, seq = texts.shape
        emb = rearrange(self.embed(texts), "bs cap seq e -> (bs cap) seq e")
        state = self.init_state(img_features, num_cap)
        out, _ = self.rnn(emb, state)
        return rearrange(out, "(bs cap) seq h -> bs cap seq h", cap=num_cap)

    def step(self, token_ids, state):
        emb = self.embed(token_ids)[:, None, :]
        out, state = self.rnn(emb, state)
        return out[:, 0], state


class ImageCaptioningModel(nn.Module):
    def __init__(self, vocab_size, hidden_size=512, num_layers=2,
                 dropout=0.3, freeze_cnn=True, pad_id=3):
        super().__init__()
        self.encoder = ImageEncoder(freeze=freeze_cnn)
        self.decoder = TextDecoder(vocab_size, self.encoder.out_features,
                                   hidden_size=hidden_size, num_layers=num_layers,
                                   dropout=dropout, pad_id=pad_id)
        self.head = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(hidden_size, vocab_size),
            nn.LogSoftmax(dim=-1),
        )

    def forward(self, imgs, texts):
        feats = self.encoder(imgs)
        hidden = self.decoder(texts, feats)
        return self.head(hidden)
