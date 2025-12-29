import math
from typing import Callable, Optional, Tuple, Union

import einops
import torch
from torch import nn
from torch.nn import functional as F

import utils

class Attention(nn.Module):
    """Multihead attention.

    Adapted from timm's ViT implementation.
    """

    def __init__(
        self,
        dim: int,
        num_heads: int,
        kdim: Optional[int] = None,
        vdim: Optional[int] = None,
        inner_dim: Optional[int] = None,
        qkv_bias: bool = False,
        attn_drop: float = 0.0,
        proj_drop: float = 0.0,
    ):
        super().__init__()
        kdim = dim if kdim is None else kdim
        vdim = dim if vdim is None else vdim
        inner_dim = dim if inner_dim is None else inner_dim
        if inner_dim % num_heads != 0:
            raise ValueError("`inner_dim` must be divisible by `num_heads`")

        self.num_heads = num_heads
        self.inner_dim = inner_dim
        self.head_dim = inner_dim // num_heads
        self.scale = self.head_dim**-0.5

        self._same_qkv_dim = dim == kdim and dim == vdim
        self._same_kv_dim = kdim == vdim

        if self._same_qkv_dim:
            self.qkv = nn.Linear(dim, inner_dim * 3, bias=qkv_bias)
        elif self._same_kv_dim:
            self.q = nn.Linear(dim, inner_dim, bias=qkv_bias)
            self.kv = nn.Linear(kdim, inner_dim * 2, bias=qkv_bias)
        else:
            self.q = nn.Linear(dim, inner_dim, bias=qkv_bias)
            self.k = nn.Linear(kdim, inner_dim, bias=qkv_bias)
            self.v = nn.Linear(vdim, inner_dim, bias=qkv_bias)

        self.attn_drop = nn.Dropout(attn_drop)
        self.out_proj = nn.Linear(inner_dim, dim)
        self.out_proj_drop = nn.Dropout(proj_drop)

        self.init_parameters()

    def init_parameters(self):
        if self._same_qkv_dim:
            bound = math.sqrt(6.0 / (self.qkv.weight.shape[0] // 3 + self.qkv.weight.shape[1]))
            nn.init.uniform_(self.qkv.weight, -bound, bound)  # Xavier init for separate Q, K, V
            if self.qkv.bias is not None:
                nn.init.constant_(self.qkv.bias, 0.0)
        elif self._same_kv_dim:
            utils.init_parameters(self.q, "xavier_uniform")
            bound = math.sqrt(6.0 / (self.kv.weight.shape[0] // 2 + self.kv.weight.shape[1]))
            nn.init.uniform_(self.kv.weight, -bound, bound)  # Xavier init for separate K, V
            if self.kv.bias is not None:
                nn.init.constant_(self.kv.bias, 0.0)
        else:
            utils.init_parameters((self.q, self.k, self.v), "xavier_uniform")

        utils.init_parameters(self.out_proj, "xavier_uniform")

    def _in_proj(self, q, k, v):
        """Efficiently compute in-projection.

        Adapted from torch.nn.functional.multi_head_attention.
        """
        if self._same_qkv_dim:
            w_kv = b_kv = b_q = b_k = b_v = None
            w = self.qkv.weight
            b = self.qkv.bias if hasattr(self.qkv, "bias") else None
        elif self._same_kv_dim:
            w = b = b_k = b_v = None
            w_q = self.q.weight
            w_kv = self.kv.weight
            b_q = self.q.bias if hasattr(self.q, "bias") else None
            b_kv = self.kv.bias if hasattr(self.kv, "bias") else None
        else:
            w = w_kv = b = b_kv = None
            w_q = self.q.weight
            w_k = self.k.weight
            w_v = self.v.weight
            b_q = self.q.bias if hasattr(self.q, "bias") else None
            b_k = self.k.bias if hasattr(self.k, "bias") else None
            b_v = self.v.bias if hasattr(self.v, "bias") else None

        if k is v:
            if q is k:
                # Self-attention
                return F.linear(q, w, b).chunk(3, dim=-1)
            else:
                # Encoder-decoder attention
                if w is not None:
                    dim = w.shape[0] // 3
                    w_q, w_kv = w.split([dim, dim * 2])
                    if b is not None:
                        b_q, b_kv = b.split([dim, dim * 2])
                return (F.linear(q, w_q, b_q),) + F.linear(k, w_kv, b_kv).chunk(2, dim=-1)
        else:
            if w is not None:
                w_q, w_k, w_v = w.chunk(3)
                if b is not None:
                    b_q, b_k, b_v = b.chunk(3)
            elif w_kv is not None:
                w_k, w_v = w_kv.chunk(2)
                if b_kv is not None:
                    b_k, b_v = b_kv.chunk(2)

            return F.linear(q, w_q, b_q), F.linear(k, w_k, b_k), F.linear(v, w_v, b_v)

    def forward(
        self,
        query: torch.Tensor,
        key: Optional[torch.Tensor] = None,
        value: Optional[torch.Tensor] = None,
        attn_mask: Optional[torch.Tensor] = None,
        key_padding_mask: Optional[torch.Tensor] = None,
        return_weights: bool = False,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        key = key if key is not None else query
        value = value if value is not None else query

        bs, n_queries, _ = query.shape
        n_keys = key.shape[1]

        if attn_mask is not None:
            if attn_mask.ndim == 2:
                expected = (n_queries, n_keys)
                if attn_mask.shape != expected:
                    raise ValueError(
                        f"2D `attn_mask` should have shape {expected}, but has "
                        f"shape {attn_mask.shape}"
                    )
                attn_mask = attn_mask.unsqueeze(0)
            elif attn_mask.ndim == 3:
                expected = (bs * self.num_heads, n_queries, n_keys)
                if attn_mask.shape != expected:
                    raise ValueError(
                        f"3D `attn_mask` should have shape {expected}, but has "
                        f"shape {attn_mask.shape}"
                    )
        if key_padding_mask is not None:
            assert key_padding_mask.dtype == torch.bool
            expected = (bs, n_keys)
            if key_padding_mask.shape != expected:
                raise ValueError(
                    f"`key_padding_mask` should have shape {expected}, but has shape "
                    f"{key_padding_mask.shape}"
                )
            key_padding_mask = einops.repeat(
                key_padding_mask, "b n -> (b h) 1 n", b=bs, h=self.num_heads, n=n_keys
            )
            if attn_mask is None:
                attn_mask = key_padding_mask
            else:
                attn_mask = attn_mask.masked_fill(key_padding_mask, float("-inf"))

        q, k, v = self._in_proj(query, key, value)

        q = einops.rearrange(q, "b n (h d) -> (b h) n d", h=self.num_heads, d=self.head_dim)
        k = einops.rearrange(k, "b n (h d) -> (b h) n d", h=self.num_heads, d=self.head_dim)
        v = einops.rearrange(v, "b n (h d) -> (b h) n d", h=self.num_heads, d=self.head_dim)

        q_scaled = q / self.scale
        if attn_mask is not None:
            attn = torch.baddbmm(attn_mask, q_scaled, k.transpose(-2, -1))
        else:
            attn = torch.bmm(q_scaled, k.transpose(-2, -1))

        attn = attn.softmax(dim=-1)  # (B x H) x N x M
        pre_dropout_attn = attn
        attn = self.attn_drop(attn)

        weighted_v = attn @ v
        x = einops.rearrange(weighted_v, "(b h) n d -> b n (h d)", h=self.num_heads, d=self.head_dim)
        x = self.out_proj(x)
        x = self.out_proj_drop(x)

        if return_weights:
            weights = einops.rearrange(pre_dropout_attn, "(b h) n m -> b h n m", h=self.num_heads)
            return x, weights.mean(dim=1)
        else:
            return x, None


class TransformerEncoderLayer(nn.TransformerEncoderLayer):
    """Like torch.nn.TransformerEncoderLayer, but with customizations."""

    def __init__(
        self,
        d_model: int,
        nhead: int,
        dim_feedforward: int = 2048,
        dim_attn: Optional[int] = None,
        dim_kv: Optional[int] = None,
        qkv_bias: bool = True,
        dropout: float = 0.1,
        activation: Union[str, Callable[[torch.Tensor], torch.Tensor]] = torch.nn.functional.relu,
        layer_norm_eps: float = 1e-5,
        batch_first: bool = False,
        norm_first: bool = False,
        initial_residual_scale: Optional[float] = None,
        device=None,
        dtype=None,
    ):
        super().__init__(
            d_model,
            nhead,
            dim_feedforward,
            dropout,
            activation,
            layer_norm_eps,
            batch_first,
            norm_first,
            device=device,
            dtype=dtype,
        )
        self.self_attn = Attention(
            dim=d_model,
            num_heads=nhead,
            kdim=dim_kv,
            vdim=dim_kv,
            inner_dim=dim_attn,
            qkv_bias=qkv_bias,
            attn_drop=dropout,
            proj_drop=dropout,
        )

        if initial_residual_scale is not None:
            self.scale1 = utils.LayerScale(d_model, init_values=initial_residual_scale)
            self.scale2 = utils.LayerScale(d_model, init_values=initial_residual_scale)
        else:
            self.scale1 = nn.Identity()
            self.scale2 = nn.Identity()

    def _sa_block(
        self,
        x: torch.Tensor,
        attn_mask: Optional[torch.Tensor] = None,
        key_padding_mask: Optional[torch.Tensor] = None,
        keys: Optional[torch.Tensor] = None,
        values: Optional[torch.Tensor] = None,
        return_weights: bool = False,
    ) -> torch.Tensor:
        keys = keys if keys is not None else x
        values = values if values is not None else x
        x, attn = self.self_attn(
            x,
            keys,
            values,
            attn_mask=attn_mask,
            key_padding_mask=key_padding_mask,
            return_weights=return_weights,
        )
        x = self.dropout1(x)

        if return_weights:
            return x, attn
        else:
            return x

    def forward(
        self,
        src: torch.Tensor,
        src_mask: Optional[torch.Tensor] = None,
        src_key_padding_mask: Optional[torch.Tensor] = None,
        memory: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        x = src
        if self.norm_first:
            x = x + self.scale1(
                self._sa_block(
                    self.norm1(x), src_mask, src_key_padding_mask, keys=memory, values=memory
                )
            )
            x = x + self.scale2(self._ff_block(self.norm2(x)))
        else:
            x = self.norm1(
                x
                + self.scale1(
                    self._sa_block(x, src_mask, src_key_padding_mask, keys=memory, values=memory)
                )
            )
            x = self.norm2(x + self.scale2(self._ff_block(x)))

        return x


class TransformerEncoder(nn.Module):
    def __init__(
        self,
        dim: int,
        n_blocks: int,
        n_heads: int,
        qkv_dim: Optional[int] = None,
        memory_dim: Optional[int] = None,
        qkv_bias: bool = True,
        dropout: float = 0.0,
        activation: Union[str, Callable[[torch.Tensor], torch.Tensor]] = "relu",
        hidden_dim: Optional[int] = None,
        initial_residual_scale: Optional[float] = None,
        frozen: bool = False,
    ):
        super().__init__()

        if hidden_dim is None:
            hidden_dim = 4 * dim

        self.blocks = nn.ModuleList(
            [
                TransformerEncoderLayer(
                    dim,
                    n_heads,
                    dim_feedforward=hidden_dim,
                    dim_attn=qkv_dim,
                    dim_kv=memory_dim,
                    qkv_bias=qkv_bias,
                    dropout=dropout,
                    activation=activation,
                    layer_norm_eps=1e-05,
                    batch_first=True,
                    norm_first=True,
                    initial_residual_scale=initial_residual_scale,
                )
                for _ in range(n_blocks)
            ]
        )

        if frozen:
            for param in self.parameters():
                param.requires_grad = False

    def forward(
        self,
        inp: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
        key_padding_mask: Optional[torch.Tensor] = None,
        memory: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        x = inp

        for block in self.blocks:
            x = block(x, mask, key_padding_mask, memory)

        return x