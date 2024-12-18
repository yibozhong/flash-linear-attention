# -*- coding: utf-8 -*-

from __future__ import annotations

from typing import Optional

from transformers.configuration_utils import PretrainedConfig


class RetNetConfig(PretrainedConfig):

    model_type = 'retnet'
    keys_to_ignore_at_inference = ['past_key_values']

    def __init__(
        self,
        vocab_size: int = 32000,
        hidden_size: int = 2048,
        expand_k: int = 1,
        expand_v: int = 2,
        hidden_ratio: Optional[int] = 2,
        intermediate_size: Optional[int] = None,
        num_hidden_layers: int = 24,
        num_heads: int = 8,
        num_kv_heads: Optional[int] = None,
        feature_map: Optional[str] = None,
        attn_mode: str = "chunk",
        hidden_act: str = "swish",
        use_short_conv: bool = False,
        conv_size: int = 4,
        use_output_gate: bool = True,
        max_position_embeddings: int = 2048,
        elementwise_affine: Optional[bool] = True,
        norm_eps: float = 1e-6,
        use_cache: bool = True,
        pad_token_id: int = None,
        bos_token_id: int = 1,
        eos_token_id: int = 2,
        tie_word_embeddings: bool = False,
        initializer_range: float = 0.02,
        fuse_norm: bool = True,
        fuse_cross_entropy: bool = True,
        vision: bool = False,
        class_size: int = 100,
        image_size: int = 224,
        patch_size: int = 16,
        num_channels: int = 3,
        hidden_dropout_prob: float = 0.1,
        video: bool = False,
        **kwargs
    ) -> RetNetConfig:
        self.vocab_size = vocab_size
        self.max_position_embeddings = max_position_embeddings
        self.hidden_size = hidden_size
        self.expand_k = expand_k
        self.expand_v = expand_v
        self.hidden_ratio = hidden_ratio
        self.intermediate_size = intermediate_size
        self.num_hidden_layers = num_hidden_layers
        self.num_heads = num_heads
        self.num_kv_heads = num_kv_heads
        self.feature_map = feature_map
        self.attn_mode = attn_mode
        self.hidden_act = hidden_act
        self.use_short_conv = use_short_conv
        self.conv_size = conv_size
        self.use_output_gate = use_output_gate
        self.elementwise_affine = elementwise_affine
        self.norm_eps = norm_eps
        self.use_cache = use_cache
        self.initializer_range = initializer_range
        self.fuse_norm = fuse_norm
        self.fuse_cross_entropy = fuse_cross_entropy
        # vision model settings
        self.vision = vision # whether apply it as a vision model
        self.class_size = class_size
        self.image_size = image_size
        self.patch_size = patch_size
        self.num_channels = num_channels
        self.hidden_dropout_prob = hidden_dropout_prob
        # video model settings
        self.video = video # whether apply it as a video model
        

        super().__init__(
            pad_token_id=pad_token_id,
            bos_token_id=bos_token_id,
            eos_token_id=eos_token_id,
            tie_word_embeddings=tie_word_embeddings,
            **kwargs,
        )
