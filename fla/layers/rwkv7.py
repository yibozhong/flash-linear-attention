# -*- coding: utf-8 -*-
# Copyright (c) 2023-2025, Songlin Yang, Yu Zhang

from __future__ import annotations

from typing import TYPE_CHECKING, Optional, Tuple

import torch
import torch.nn as nn
from einops import rearrange
from torch.nn import functional as F

from fla.layers.rwkv6 import LoRA
from fla.modules import GroupNorm
from fla.modules.l2norm import l2_norm
from fla.ops.rwkv7 import chunk_rwkv7, fused_recurrent_rwkv7
from fla.ops.rwkv7.fused_addcmul import fused_addcmul_rwkv7

if TYPE_CHECKING:
    from fla.models.utils import Cache


class RWKV7Attention(nn.Module):

    def __init__(
        self,
        mode: str = 'chunk',
        hidden_size: int = 1024,
        head_dim: Optional[int] = 64,
        num_heads: Optional[int] = None,
        decay_low_rank_dim: int = 64,
        gate_low_rank_dim: int = 128,
        a_low_rank_dim: int = 64,
        v_low_rank_dim: int = 16,
        elementwise_affine: Optional[bool] = True,
        norm_eps: float = 1e-5,
        layer_idx: int = None,
        fuse_norm: bool = False,
        value_dim: int = None,
        **kwargs
    ) -> RWKV7Attention:
        super().__init__()

        self.mode = mode
        assert mode in ['chunk', 'fused_recurrent'], f"Not supported mode `{mode}`."
        self.hidden_size = hidden_size

        self.key_dim = hidden_size
        self.value_dim = value_dim if value_dim is not None else hidden_size
        if head_dim is None and num_heads is None:
            raise ValueError("Either `head_dim` or `num_heads` must be specified.")
        elif head_dim is not None:
            self.head_dim = head_dim
            self.num_heads = int(hidden_size // head_dim)
        elif num_heads is not None:
            self.head_dim = int(hidden_size // num_heads)
            self.num_heads = num_heads
        self.head_v_dim = int(self.value_dim // self.num_heads)

        self.decay_low_rank_dim = decay_low_rank_dim
        self.gate_low_rank_dim = gate_low_rank_dim
        self.a_low_rank_dim = a_low_rank_dim
        self.v_low_rank_dim = v_low_rank_dim
        self.layer_idx = layer_idx
        self.fuse_norm = fuse_norm

        self.time_shift = nn.ZeroPad2d((0, 0, 1, -1))
        self.x_r = nn.Parameter(torch.zeros(1, 1, hidden_size))
        self.x_w = nn.Parameter(torch.zeros(1, 1, hidden_size))
        self.x_k = nn.Parameter(torch.zeros(1, 1, hidden_size))
        self.x_v = nn.Parameter(torch.zeros(1, 1, hidden_size))
        self.x_a = nn.Parameter(torch.zeros(1, 1, hidden_size))
        self.x_g = nn.Parameter(torch.zeros(1, 1, hidden_size))

        self.k_k = nn.Parameter(torch.zeros(self.key_dim))
        self.k_a = nn.Parameter(torch.zeros(self.key_dim))
        self.r_k = nn.Parameter(torch.zeros(self.num_heads, self.head_dim))

        self.r_proj = nn.Linear(hidden_size, self.key_dim, bias=False)
        self.k_proj = nn.Linear(hidden_size, self.key_dim, bias=False)
        self.v_proj = nn.Linear(hidden_size, self.value_dim, bias=False)
        self.o_proj = nn.Linear(self.value_dim, hidden_size, bias=False)

        self.w_lora = LoRA(hidden_size, self.key_dim, low_rank_dim=decay_low_rank_dim, activation='tanh')
        if self.layer_idx != 0:
            self.v_lora = LoRA(hidden_size, self.value_dim, low_rank_dim=v_low_rank_dim, activation=None)
        self.a_lora = LoRA(hidden_size, self.key_dim, low_rank_dim=a_low_rank_dim, activation=None)
        self.g_lora = LoRA(hidden_size, self.value_dim, low_rank_dim=gate_low_rank_dim, activation='sigmoid', bias=False)

        if self.fuse_norm:
            self.g_norm = GroupNorm(
                num_groups=self.num_heads,
                hidden_size=self.value_dim,
                elementwise_affine=elementwise_affine,
                eps=self.head_dim*norm_eps,
                bias=True,
            )
        else:
            self.g_norm = nn.GroupNorm(
                num_groups=self.num_heads,
                num_channels=self.value_dim,
                eps=self.head_dim*norm_eps,
                affine=elementwise_affine
            )

        self.apply(self._initialize_weights)

    def _initialize_weights(self, module: nn.Module):
        if getattr(module, "_is_hf_initialized", False):
            return
        if isinstance(module, nn.Linear):
            nn.init.xavier_uniform_(module.weight, gain=2 ** -2.5)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        if isinstance(module, nn.Parameter):
            nn.init.xavier_uniform_(module, gain=2 ** -2.5)
        module._is_hf_initialized = True

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        past_key_values: Optional[Cache] = None,
        use_cache: Optional[bool] = False,
        output_attentions: Optional[bool] = False,
        v_first: torch.Tensor = None,
        **kwargs
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[Cache]]:
        if attention_mask is not None:
            assert len(attention_mask.shape) == 2, (
                "Expected attention_mask as a 0-1 matrix with shape [batch_size, seq_len] "
                "for padding purposes (0 indicating padding). "
                "Arbitrary attention masks of shape [batch_size, seq_len, seq_len] are not allowed."
            )

        batch_size, seq_len, _ = hidden_states.shape

        if self.training:
            # if training, use chunk mode no matter how short the sequence is
            mode = 'chunk'
        else:
            # launching the triton kernel for just one token will actually be slower
            mode = 'fused_recurrent' if hidden_states.shape[1] <= 64 else self.mode

        last_state = None
        if past_key_values is not None and len(past_key_values) > self.layer_idx:
            last_state = past_key_values[self.layer_idx]

        if attention_mask is not None:
            hidden_states = hidden_states.mul(attention_mask[:, -hidden_states.shape[-2]:, None])
        if hidden_states.shape[1] == 1 and last_state is not None:
            shifted = last_state['conv_state'].unsqueeze(1)
        else:
            shifted = self.time_shift(hidden_states)
            if last_state is not None:
                shifted[:, 0] = last_state['conv_state']

        # [batch_size, seq_len, hidden_size]
        delta = shifted - hidden_states

        xr, xw, xk, xv, xa, xg = fused_addcmul_rwkv7(hidden_states, delta, self.x_r, self.x_w,
                                                     self.x_k, self.x_v, self.x_a, self.x_g)

        r = self.r_proj(xr)
        # w (-0.6065, 0)
        # when we apply sigmoid, bf16 input will not have numerical issue
        # (w.float() - w2).abs().max()/mean() = 0.003, 0.0004
        w = -0.6065306597126334 * self.w_lora(xw).sigmoid()

        k = self.k_proj(xk)
        v = self.v_proj(xv)

        if self.layer_idx == 0:
            v_first = v
        else:
            v = torch.lerp(v, v_first, self.v_lora(xv).sigmoid())
        a = self.a_lora(xa).sigmoid()
        g = self.g_lora(xg)

        if self.fuse_norm:
            kk = l2_norm(rearrange(k * self.k_k, 'b t (h d) -> b t h d', d=self.head_dim))
        else:
            kk = F.normalize(rearrange(k * self.k_k, 'b t (h d) -> b t h d', d=self.head_dim), dim=-1, p=2.0)

        k = k.addcmul(k * (a - 1), self.k_a)

        # dealing with left-padding
        if attention_mask is not None:
            v = v * attention_mask[:, -v.shape[-2]:, None]
        r, w, k, a = map(lambda x: rearrange(x, 'b t (h d) -> b t h d', d=self.head_dim), (r, w, k, a))
        v = rearrange(v, 'b t (h d) -> b t h d', d=self.head_v_dim)

        recurrent_state = last_state['recurrent_state'] if last_state is not None else None

        rwkv7_fn = chunk_rwkv7 if mode == 'chunk' else fused_recurrent_rwkv7
        cu_seqlens = kwargs.get('cu_seqlens', None)
        o, recurrent_state = rwkv7_fn(
            r=r,
            w=w,
            k=k,
            v=v,
            a=-kk,
            b=kk * a,
            scale=1.,
            initial_state=recurrent_state,
            output_final_state=use_cache,
            cu_seqlens=cu_seqlens,
            head_first=False,
            input_precision=r.dtype,
        )

        if past_key_values is not None:
            past_key_values.update(
                recurrent_state=recurrent_state,
                conv_state=hidden_states[:, -1],
                layer_idx=self.layer_idx,
                offset=r.shape[1]
            )

        if self.fuse_norm:
            o = self.g_norm(rearrange(o, '... h d -> ... (h d)'))
        else:
            o = self.g_norm(rearrange(o, 'b t h d -> (b t) (h d)')).view(batch_size, seq_len, -1)

        o = o + ((r * k * self.r_k).sum(-1, keepdim=True) * v).view(batch_size, seq_len, -1)
        o = self.o_proj(o * g)

        return o, None, past_key_values, v_first
