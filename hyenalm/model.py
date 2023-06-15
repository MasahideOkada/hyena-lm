from dataclasses import dataclass, asdict
from typing import Any, Callable

import torch
from torch import Tensor
import torch.nn as nn
import torch.nn.functional as fn

from modules import HyenaBlock

@dataclass(frozen=True)
class HyenaConfig:
    embed_dim: int = 512
    max_seq_len: int = 256
    order: int = 2
    pos_dim: int = 65
    kernel_size: int = 3
    stride: int = 1
    padding: int = 2
    ffn_depth: int = 4
    ffn_hidden_size: int = 64
    freq: float = 8.0
    learn_filter: bool = True
    fast_decay_t: float = 0.3
    slow_decay_t: float = 1.5
    target: float = 1e-2
    shift: float = 0.0
    activation: str = "identity"

class HyenaLM(nn.Module):
    def __init__(
        self,
        vocab_size: int,
        depth: int,
        hyena_config: HyenaConfig,
        p_dropout: float = 0.0,
        pe_type: str = "absolute",
        pad_id: int = 0
    ):
        super().__init__()
        self.vocab_size = vocab_size
        embed_dim = hyena_config.embed_dim
        self.embed = nn.Embedding(vocab_size, embed_dim, padding_idx=pad_id)
        self.dropout = nn.Dropout(p_dropout)

        pos_embed: Tensor
        pe_requires_grad = True
        match name := pe_type.lower():
            case "absolute":
                pos_embed = torch.randn(1, hyena_config.max_seq_len, embed_dim)
            case "nope":
                pos_embed = torch.zeros(1, hyena_config.max_seq_len, embed_dim)
                pe_requires_grad = False
            case _: raise NotImplementedError(f"positional encoding `{name}` is invalid")
        self.pos_embed = nn.Parameter(pos_embed, requires_grad=pe_requires_grad)

        self.layers = nn.Sequential()
        for _ in range(depth):
            self.layers.append(HyenaBlock(**asdict(hyena_config)))
        self.norm = nn.LayerNorm(embed_dim)
        self.out = nn.Linear(embed_dim, vocab_size)

    def forward(self, x: Tensor) -> Tensor:
        # B: batch size, L: seq_len, E: embed dim, V: vocab size
        x = self.embed(x) # (B, L) -> (B, L, E)
        x += self.pos_embed[:, :x.shape[1], :]
        x = self.dropout(x) # (B, L, E) -> (B, L, E)
        x = self.layers(x) # (B, L, E) -> (B, L, E)
        x = self.norm(x) # (B, L, E) -> (B, L, E)
        x = self.out(x) # (B, L, E) -> (B, L, V)
        return x

def generate(
    model: HyenaLM,
    prompt: str,
    output_len: int,
    encoder: Callable[[str], list[int]],
    decoder: Callable[[list[int]], str],
    max_seq_len: int,
    bos: int,
    eos: int,
    k: int = 10,
    temperature: float = 1.0,
    device: Any = torch.device("cpu")
) -> str:
    # L: seq len, V: vocab size, K: k
    model.eval()
    model.to(device=device)
    tokens = encoder(prompt)
    tokens.insert(0, bos)
    while len(tokens) < output_len + 1:
        x = torch.unsqueeze(
            torch.tensor(tokens[-max_seq_len:], dtype=torch.long, device=device), 0
        ) # -> (1, L)
        logits = model(x)[0, -1, :] # -> (V)

        values, indices = torch.topk(logits, k) # -> (K) for each
        probas = torch.full_like(logits, float("-inf")) # -> (V)
        probas.scatter_(0, indices, values)
        probas = fn.softmax(probas / temperature, dim=-1) # (V) -> (V)
        next_token = torch.multinomial(probas, 1).item()

        if next_token == eos: break
        tokens.append(next_token)
    output = decoder(tokens)
    return output