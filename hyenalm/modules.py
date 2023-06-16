import numpy as np

import torch
from torch import Tensor
import torch.nn as nn

class Projection(nn.Module):
    def __init__(
        self,
        embed_dim: int,
        order: int = 2,
        kernel_size: int = 3,
        stride: int = 1,
        padding: int = 2
    ):
        super().__init__()
        hidden_size = (order + 1) * embed_dim
        self.linear = nn.Linear(embed_dim, hidden_size)
        self.short_conv = nn.Conv1d(
            hidden_size,
            hidden_size,
            kernel_size,
            stride=stride,
            padding=padding,
            groups=hidden_size
        )
        self.hidden_size = hidden_size

    def forward(self, x: Tensor) -> list[Tensor]:
        """
        args
        - `x`: input tensor with shape (batches, length, embed dim)
        """
        # B: batch size, L: seq len, E: embed dim, N: order of hyena
        L = x.shape[1]
        x = (
            self.linear(x) # (B, L, E) -> (B, L, (N+1)*E)
            .transpose(1, 2) # (B, L, (N+1)*E) -> (B, (N+1)*E, L)
        )
        x = self.short_conv(x)[..., :L] # (B, (N+1)*E, L) -> (B, (N+1)*E, L)
        return x.chunk(self.hidden_size, dim=1) # (B, (N+1)*E, L) -> [(B, E, L)] * (N+1)

class PositionalEncoding(nn.Module):
    def __init__(self, embed_dim: int, max_seq_len: int):
        assert embed_dim % 2 == 1, "`embed_dim` must be odd"
        super().__init__()
        # L: seq len, Ep: pos embed dim, K: (Et-1)//2
        t = torch.linspace(0, 1, steps=max_seq_len).unsqueeze(-1) # -> (L, 1)
        t_pos = torch.arange(0, max_seq_len, dtype=torch.float).unsqueeze(-1) # -> (L, 1)
        K = (embed_dim - 1) // 2
        k = torch.linspace(0, K - 1, steps=K).unsqueeze(0) # -> (1, K)
        z = torch.exp(1j * 2 * np.pi * k * t_pos / max_seq_len) # -> (L, K)
        self.t = nn.Parameter(t.view(1, 1, max_seq_len), requires_grad=False) # -> (1, 1, L)
        self.z = nn.Parameter(
            torch.cat([t, z.real, z.imag], dim=-1), # -> (L, Ep)
        )

    def forward(self, seq_len: int) -> tuple[Tensor, Tensor]:
        return self.t[..., :seq_len], self.z[:seq_len, :]

class Sin(nn.Module):
    def __init__(self, embed_dim: int, freq: float = 8.0, learn: bool = True):
        super().__init__()
        self.freq = nn.Parameter(freq * torch.ones(1, embed_dim), requires_grad=learn)

    def forward(self, x: Tensor) -> Tensor:
        # L: seq len, E: embed dim
        return torch.sin(self.freq * x) # -> (L, E)

class ExponentialDecayWindow(nn.Module):
    def __init__(
        self,
        embed_dim: int,
        fast_decay_t: float = 0.3,
        slow_decay_t: float = 1.5,
        target: float = 1e-2,
        shift: float = 0.0
    ):
        super().__init__()
        max_decay = np.log(target) / fast_decay_t
        min_decay = np.log(target) / slow_decay_t
        self.alphas = nn.Parameter(
            torch.linspace(min_decay, max_decay, steps=embed_dim).view(1, embed_dim, 1)
        )
        self.shift = shift

    def forward(self, x: Tensor, t: Tensor) -> Tensor:
        # L: seq len, E: embed dim, N: order of hyena
        L = x.shape[-1]
        decay = torch.exp(self.alphas * t)[..., :L] # -> (1, E, L)
        x *= (decay + self.shift)
        return x

class HyenaFilter(nn.Module):
    def __init__(
        self,
        pos_embed_dim: int,
        max_seq_len: int,
        seq_embed_dim: int,
        order: int = 2,
        ffn_depth: int = 4,
        ffn_hidden_size: int = 64,
        freq: float = 10.0,
        learn: bool = True,
        fast_decay_t: float = 0.3,
        slow_decay_t: float = 1.5,
        target: float = 1e-2,
        shift: float = 0.0
    ):
        assert ffn_depth > 2, "`ffn_depth` must be greater than 2"
        super().__init__()
        self.pos = PositionalEncoding(pos_embed_dim, max_seq_len)

        self.ffn = nn.Sequential(
            nn.Linear(pos_embed_dim, ffn_hidden_size),
            Sin(ffn_hidden_size, freq, learn)
        )
        for _ in range(ffn_depth - 2):
            self.ffn.append(nn.Linear(ffn_hidden_size, ffn_hidden_size))
            self.ffn.append(Sin(ffn_hidden_size, freq, learn))
        self.ffn.append(nn.Linear(ffn_hidden_size, order * seq_embed_dim, bias=False))

        self.embed_dim = seq_embed_dim
        self.order = order
        self.window = ExponentialDecayWindow(
            seq_embed_dim,
            fast_decay_t=fast_decay_t,
            slow_decay_t=slow_decay_t,
            target=target,
            shift=shift
        )

    def forward(self, seq_len: int) -> list[Tensor]:
        # L: seq len, Ep: pos embed dim, N: order of hyena, E: seq embed dim
        t, z = self.pos(seq_len) # -> (1, 1, L), (L, Ep)
        h = (
            self.ffn(z) # (L, Ep) -> (L, N*E)
            .transpose(0, 1) # (L, N*E) -> (N*E, L)
            .reshape(self.order, self.embed_dim, seq_len) # (N*E, L) -> (N, E, L)
        )
        h = self.window(h, t) # (N, E, L) -> (N, E, L)
        return h.chunk(self.order, dim=0) # (N, E, L) -> [(1, E, L)] * N

class HyenaBlock(nn.Module):
    def __init__(
        self,
        *,
        embed_dim: int,
        max_seq_len: int,
        order: int,
        pos_dim: int = 65,
        kernel_size: int = 3,
        stride: int = 1,
        padding: int = 2,
        ffn_depth: int = 4,
        ffn_hidden_size: int = 64,
        freq: float = 8.0,
        learn_filter: bool = True,
        fast_decay_t: float = 0.3,
        slow_decay_t: float = 1.5,
        target: float = 1e-2,
        shift: float = 0.0,
        activation: str = "identity"
    ):
        super().__init__()
        self.proj = Projection(
            embed_dim,
            order,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding
        )
        self.hyena_filter = HyenaFilter(
            pos_dim,
            max_seq_len,
            seq_embed_dim=embed_dim,
            order=order,
            ffn_depth=ffn_depth,
            ffn_hidden_size=ffn_hidden_size,
            freq=freq,
            learn=learn_filter,
            fast_decay_t=fast_decay_t,
            slow_decay_t=slow_decay_t,
            target=target,
            shift=shift
        )
        self.bias = nn.Parameter(torch.randn(order, 1, embed_dim, 1))
        self.norm1 = nn.LayerNorm(embed_dim)
        self.norm2 = nn.LayerNorm(embed_dim)
        self.fc = nn.Linear(embed_dim, embed_dim)

        act: nn.Module
        match name := activation.lower():
            case "identity": act = nn.Identity()
            case "relu": act = nn.ReLU()
            case "leaky-relu": act = nn.LeakyReLU()
            case "gelu": act = nn.GELU()
            case "silu": act = nn.SiLU()
            case "tanh": act = nn.Tanh()
            case _: raise NotImplementedError(f"activation `{name}` is invalid")
        self.act = act

    @staticmethod
    def fftconv(x: Tensor, h: Tensor, d: Tensor) -> Tensor:
        # B: batch size, L: seq len, E: embed dim
        L = x.shape[-1]
        h_fft = torch.fft.rfft(h, n=2*L, norm="forward") # (1, E, L) -> (1, E, 2*L)
        x_fft = torch.fft.rfft(x.to(dtype=h.dtype), n=2*L) # (B, E, L) -> (B, E, 2*L)
        y = torch.fft.irfft(x_fft * h_fft, n=2*L, norm="forward")[..., :L] # -> (B, E, L)
        y += x * d
        return y.to(dtype=x.dtype)

    def forward(self, u: Tensor) -> Tensor:
        # B: batch size, L: seq len, E: embed dim, N: order of hyena
        L = u.shape[1]
        x = self.norm1(u) # (B, L, E) -> (B, L, E)
        x = self.proj(x) # (B, L, E) -> [(B, E, L)] * (N+1)
        h = self.hyena_filter(L) # -> [(1, E, L)] * N
        v = x[-1] # -> (B, E, L)
        for x_i, h_i, d_i in zip(x[:-1], h, self.bias):
            v = x_i * self.fftconv(v, h_i, d_i)
        y = u + v.transpose(1, 2) # -> (B, L, E)
        out = self.norm2(y) # (B, L, E) -> (B, L, E)
        out = self.fc(out) # (B, L, E) -> (B, L, E)
        out = self.act(out) # (B, L, E) -> (B, L, E)
        out += y
        return out
