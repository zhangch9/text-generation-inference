# coding=utf-8
# TODO: revise copyright statement
""" Pytorch ChatGLM model"""


from collections.abc import Sequence
from functools import partial
from typing import Callable, List, Optional, Tuple, Union

import torch
import torch.distributed
from accelerate import init_empty_weights
from torch import nn
from torch.nn import functional as F
from transformers.configuration_utils import PretrainedConfig

from text_generation_server.utils import Weights, flash_attn, paged_attention
from text_generation_server.utils.import_utils import IS_CUDA_SYSTEM, IS_ROCM_SYSTEM
from text_generation_server.utils.layers import (
    PositionRotaryEmbedding,
    TensorParallelColumnLinear,
    TensorParallelEmbedding,
    TensorParallelHead,
    TensorParallelRowLinear,
    get_linear,
)

if IS_CUDA_SYSTEM:
    import dropout_layer_norm
elif IS_ROCM_SYSTEM:
    from vllm import layernorm_ops


_shape_t = Union[int, Sequence[int], torch.Size]


class ChatGLMConfig(PretrainedConfig):
    model_type = "chatglm"

    def __init__(
        self,
        num_layers: int = 28,
        padding_vocab_size: int = 65024,
        hidden_size: int = 4096,
        ffn_hidden_size: int = 13696,
        kv_channels: int = 128,
        num_attention_heads: int = 32,
        seq_length: int = 2048,
        hidden_dropout: Optional[float] = 0.0,
        classifier_dropout: Optional[float] = None,
        attention_dropout: Optional[float] = 0.0,
        layernorm_epsilon: float = 1e-5,
        rope_ratio: float = 1.0,
        rmsnorm: bool = True,
        apply_residual_connection_post_layernorm: bool = False,
        post_layer_norm: bool = True,
        add_bias_linear: bool = False,
        add_qkv_bias: bool = False,
        bias_dropout_fusion: bool = True,
        multi_query_attention: bool = False,
        multi_query_group_num: int = 1,
        apply_query_key_layer_scaling: bool = True,
        attention_softmax_in_fp32: bool = True,
        fp32_residual_connection: bool = False,
        quantization_bit: int = 0,
        **kwargs,
    ):
        self.num_layers = num_layers
        self.padding_vocab_size = padding_vocab_size
        self.hidden_size = hidden_size
        self.ffn_hidden_size = ffn_hidden_size
        self.kv_channels = kv_channels
        self.num_attention_heads = num_attention_heads
        self.seq_length = seq_length
        self.hidden_dropout = hidden_dropout
        self.classifier_dropout = classifier_dropout
        self.attention_dropout = attention_dropout
        self.layernorm_epsilon = layernorm_epsilon
        self.rope_ratio = rope_ratio
        self.rmsnorm = rmsnorm
        self.apply_residual_connection_post_layernorm = (
            apply_residual_connection_post_layernorm
        )
        self.post_layer_norm = post_layer_norm
        self.add_bias_linear = add_bias_linear
        self.add_qkv_bias = add_qkv_bias
        self.bias_dropout_fusion = bias_dropout_fusion
        self.multi_query_attention = multi_query_attention
        self.multi_query_group_num = multi_query_group_num
        self.apply_query_key_layer_scaling = apply_query_key_layer_scaling
        self.attention_softmax_in_fp32 = attention_softmax_in_fp32
        self.fp32_residual_connection = fp32_residual_connection
        self.quantization_bit = quantization_bit
        super().__init__(**kwargs)


class GLMEmbedding(nn.Module):
    def __init__(self, prefix: str, config: ChatGLMConfig, weights: Weights):
        super().__init__()

        self.word_embeddings = TensorParallelEmbedding(
            prefix=f"{prefix}.word_embeddings", weights=weights
        )
        self.fp32_residual_connection = config.fp32_residual_connection

    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        embeddings = self.word_embeddings(input_ids)
        if self.fp32_residual_connection:
            embeddings = embeddings.float()
        return embeddings


class GLMRMSNorm(nn.Module):
    def __init__(
        self, normalized_shape: _shape_t, eps: float = 1e-5, device=None, dtype=None
    ):
        super().__init__()

        self.weight = torch.nn.Parameter(
            torch.empty(normalized_shape, device=device, dtype=dtype)
        )
        self.eps = eps

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        if hidden_states.shape[-1] > 8192:
            hidden_states = hidden_states.to(torch.float32)
            variance = hidden_states.pow(2).mean(-1, keepdim=True)
            hidden_states = hidden_states * torch.rsqrt(variance + self.eps)
            if self.weight.dtype in [torch.float16, torch.bfloat16]:
                hidden_states = hidden_states.to(self.weight.dtype)

            return self.weight * hidden_states

        if IS_CUDA_SYSTEM:
            ret = dropout_layer_norm.dropout_add_ln_fwd(
                hidden_states,
                None,
                self.weight,
                None,
                None,
                None,
                None,
                None,
                0.0,
                self.eps,
                1.0,
                0,
                None,
                False,
                True,
            )
            hidden_states = ret[0]
            return hidden_states

        if IS_ROCM_SYSTEM:
            out = torch.empty_like(hidden_states)
            layernorm_ops.rms_norm(
                out, hidden_states, self.weight.data, self.variance_epsilon
            )
            return out

        raise ValueError(
            "Your system seem to be not supported. Please check your install or open an issue at https://github.com/huggingface/text-generation-inference/issues with a clear reproduction."
        )

    @classmethod
    def load(cls, prefix: str, weights: Weights, eps: float = 1e-5) -> "GLMRMSNorm":
        weight = weight = weights.get_tensor(_resolve_prefix("weight", prefix))
        with init_empty_weights():
            rms_norm = cls(weight.shape, eps=eps)
        rms_norm.weight = nn.Parameter(weight)
        return rms_norm


class GLMSelfAttention(nn.Module):
    def __init__(self, prefix: str, config: ChatGLMConfig, weights: Weights):
        super().__init__()

        hidden_size = config.hidden_size
        total_num_heads = config.num_attention_heads
        total_num_kv_heads = (
            config.multi_query_group_num
            if config.multi_query_attention
            else config.num_attention_heads
        )

        self._head_size = (
            hidden_size // total_num_heads
            if config.kv_channels is None
            else config.kv_channels
        )

        num_shards = weights.process_group.size()
        if total_num_heads % num_shards != 0:
            raise ValueError(
                f"`num_heads` must be divisible by `num_shards` (got `num_heads`: {self.num_heads} and `num_shards`: {num_shards})"
            )
        self._num_heads = total_num_heads // num_shards

        if total_num_kv_heads >= num_shards:
            if total_num_kv_heads % num_shards != 0:
                raise ValueError(
                    f"`num_kv_heads` must be divisible by `num_shards` (got `num_kv_heads`: {total_num_kv_heads} and `num_shards`: {num_shards})"
                )
            self._num_kv_heads = total_num_kv_heads // num_shards
        else:
            if num_shards % total_num_kv_heads:
                raise ValueError(
                    f"`num_shards` must be divisible by `num_kv_heads` (got `num_shards`: {num_shards} and `num_kv_heads`: {total_num_kv_heads})"
                )
            self._num_kv_heads = 1

        num_groups = self.num_heads // self.num_kv_heads
        self.softmax_scale = self.head_size**-0.5
        self.kv_head_mapping = torch.arange(
            0, self.num_kv_heads, dtype=torch.int32, device=weights.device
        ).repeat_interleave(num_groups)

        # Linear tranformations before & after attention
        qkv_bias = config.add_bias_linear or config.add_qkv_bias
        self.query_key_value = load_attention(
            config,
            _resolve_prefix("query_key_value", prefix),
            weights,
            qkv_bias,
            total_num_heads,
            total_num_kv_heads,
            self.head_size,
        )
        self.dense = TensorParallelRowLinear.load(
            config,
            prefix=_resolve_prefix("dense", prefix),
            weights=weights,
            bias=config.add_bias_linear,
        )

    def forward(
        self,
        hidden_states: torch.Tensor,
        rope_fn: Callable[[torch.Tensor, torch.Tensor], None],
        cu_seqlen_prefill: Optional[torch.Tensor],
        kv_cache: Tuple[torch.Tensor, torch.Tensor],
        block_tables: torch.Tensor,
        slots: torch.Tensor,
        input_lengths: torch.Tensor,
        max_s: int,
    ) -> torch.Tensor:
        qkv = self.query_key_value(hidden_states)
        query, kv = qkv.split(
            [self.head_size * self.num_heads, 2 * self.head_size * self.num_kv_heads],
            dim=1,
        )
        query = query.view(-1, self.num_heads, self.head_size)
        kv = kv.view(-1, 2, self.num_kv_heads, self.head_size)

        rope_fn(query, torch.select(kv, dim=1, index=0))

        paged_attention.reshape_and_cache(
            kv[:, 0], kv[:, 1], kv_cache[0], kv_cache[1], slots
        )
        attention_out = torch.empty_like(query)
        if cu_seqlen_prefill is not None:
            flash_attn.attention(
                query,
                torch.select(kv, dim=1, index=0),
                torch.select(kv, dim=1, index=1),
                attention_out,
                cu_seqlen_prefill,
                max_s,
                self.softmax_scale,
            )
        else:
            paged_attention.attention(
                attention_out,
                query,
                kv_cache[0],
                kv_cache[1],
                self.kv_head_mapping,
                self.softmax_scale,
                block_tables,
                input_lengths,
                max_s,
            )
        return self.dense(attention_out.view(-1, self.num_heads * self.head_size))

    @property
    def num_heads(self) -> int:
        return self._num_heads

    @property
    def num_kv_heads(self) -> int:
        return self._num_kv_heads

    @property
    def head_size(self) -> int:
        return self._head_size


class GLMMLP(nn.Module):
    def __init__(self, prefix: str, config: ChatGLMConfig, weights: Weights):
        super().__init__()

        bias = config.add_bias_linear
        self.dense_h_to_4h = load_linear_from_packed_weights(
            config,
            prefix=_resolve_prefix("dense_h_to_4h", prefix),
            weights=weights,
            bias=bias,
            size_lst=[config.ffn_hidden_size] * 2,
        )
        self.activation_func = swiglu
        self.dense_4h_to_h = TensorParallelRowLinear.load(
            config,
            prefix=_resolve_prefix("dense_4h_to_h", prefix),
            weights=weights,
            bias=bias,
        )

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        # [s, b, 4hp]
        intermediate_parallel = self.dense_h_to_4h(hidden_states)
        intermediate_parallel = self.activation_func(intermediate_parallel)
        # [s, b, h]
        output = self.dense_4h_to_h(intermediate_parallel)
        return output


class GLMBlock(nn.Module):
    def __init__(self, prefix: str, config: ChatGLMConfig, weights: Weights):
        super().__init__()

        self.apply_residual_connection_post_layernorm = (
            config.apply_residual_connection_post_layernorm
        )

        layer_norm_cls = GLMRMSNorm if config.rmsnorm else nn.LayerNorm
        self.input_layernorm = layer_norm_cls.load(
            _resolve_prefix("input_layernorm", prefix),
            weights,
            eps=config.layernorm_epsilon,
        )
        self.self_attention = GLMSelfAttention(
            _resolve_prefix("self_attention", prefix), config, weights
        )
        self.post_attention_layernorm = layer_norm_cls.load(
            _resolve_prefix("post_attention_layernorm", prefix),
            weights,
            eps=config.layernorm_epsilon,
        )
        self.mlp = GLMMLP(_resolve_prefix("mlp", prefix), config, weights)

    def forward(
        self,
        hidden_states: torch.Tensor,
        rope_fn: Callable[[torch.Tensor, torch.Tensor], None],
        cu_seqlen_prefill: Optional[torch.Tensor],
        kv_cache: Tuple[torch.Tensor, torch.Tensor],
        block_tables: torch.Tensor,
        slots: torch.Tensor,
        input_lengths: torch.Tensor,
        max_s: int,
    ) -> torch.Tensor:
        layernorm_out = self.input_layernorm(hidden_states)
        attention_out = self.self_attention(
            layernorm_out,
            rope_fn,
            cu_seqlen_prefill,
            kv_cache,
            block_tables,
            slots,
            input_lengths,
            max_s,
        )

        if self.apply_residual_connection_post_layernorm:
            residual = layernorm_out
        else:
            residual = hidden_states

        layernorm_in = attention_out + residual
        layernorm_out = self.post_attention_layernorm(layernorm_in)
        mlp_out = self.mlp(layernorm_out)

        if self.apply_residual_connection_post_layernorm:
            residual = layernorm_out
        else:
            residual = layernorm_in
        return mlp_out + residual

    @property
    def num_kv_heads(self) -> int:
        return self.self_attention.num_kv_heads

    @property
    def head_size(self) -> int:
        return self.self_attention.head_size


class GLMTransformer(nn.Module):
    def __init__(self, prefix: str, config: ChatGLMConfig, weights: Weights):
        super().__init__()

        self.layers = nn.ModuleList(
            [
                GLMBlock(_resolve_prefix(f"layers.{idx}", prefix), config, weights)
                for idx in range(config.num_layers)
            ]
        )

        if config.post_layer_norm:
            layer_norm_cls = GLMRMSNorm if config.rmsnorm else nn.LayerNorm
            self.final_layernorm = layer_norm_cls.load(
                _resolve_prefix("final_layernorm", prefix),
                weights,
                eps=config.layernorm_epsilon,
            )
        else:
            self.final_layernorm = None

    def forward(
        self,
        hidden_states: torch.Tensor,
        rope_fn: Callable[[torch.Tensor, torch.Tensor], None],
        cu_seqlen_prefill: Optional[torch.Tensor],
        kv_cache: List[Tuple[torch.Tensor, torch.Tensor]],
        block_tables: Optional[torch.Tensor],
        slots: torch.Tensor,
        input_lengths: torch.Tensor,
        max_s: int,
    ) -> torch.Tensor:
        for idx, layer in enumerate(self.layers):
            hidden_states = layer(
                hidden_states,
                rope_fn,
                cu_seqlen_prefill,
                kv_cache[idx],
                block_tables,
                slots,
                input_lengths,
                max_s,
            )

        if self.final_layernorm is not None:
            hidden_states = self.final_layernorm(hidden_states)
        return hidden_states

    @property
    def num_layers(self) -> int:
        return len(self.layers)

    @property
    def num_kv_heads(self) -> int:
        return self.layers[0].num_kv_heads

    @property
    def head_size(self) -> int:
        return self.layers[0].head_size


class ChatGLMModel(nn.Module):
    def __init__(self, prefix: str, config: ChatGLMConfig, weights: Weights):
        super().__init__()

        process_group = weights.process_group
        self.tp_rank = process_group.rank()
        self.tp_world_size = process_group.size()

        self.embedding = GLMEmbedding(
            _resolve_prefix("embedding", prefix), config, weights
        )
        # Rotary Position Embedding
        rope_ratio = config.rope_ratio or 1.0
        rope_dim = (
            config.hidden_size // config.num_attention_heads
            if config.kv_channels is None
            else config.kv_channels
        )
        self.rotary_emb = PositionRotaryEmbedding.static(
            config, dim=rope_dim // 2, base=10000.0 * rope_ratio, device=weights.device
        )
        self.encoder = GLMTransformer(
            _resolve_prefix("encoder", prefix), config, weights
        )
        self.output_layer = TensorParallelHead.load(
            config, _resolve_prefix("output_layer", prefix), weights=weights
        )

    def forward(
        self,
        input_ids: torch.Tensor,
        position_ids: torch.Tensor,
        cu_seqlen_prefill: Optional[torch.Tensor],
        kv_cache: List[Tuple[torch.Tensor, torch.Tensor]],
        block_tables: Optional[torch.Tensor],
        slots: torch.Tensor,
        input_lengths: torch.Tensor,
        max_s: int,
    ) -> torch.Tensor:
        hidden_states = self.embedding(input_ids)
        cos, sin = self.rotary_emb.get_cos_sin(position_ids, max_s, hidden_states.dtype)
        rope_fn = partial(self.rotary_emb.forward, cos=cos, sin=sin, is_neox=False)
        hidden_states = self.encoder(
            hidden_states,
            rope_fn,
            cu_seqlen_prefill,
            kv_cache,
            block_tables,
            slots,
            input_lengths,
            max_s,
        )
        return hidden_states

    @property
    def num_layers(self) -> int:
        return self.encoder.num_layers

    @property
    def num_kv_heads(self) -> int:
        return self.encoder.num_kv_heads

    @property
    def head_size(self) -> int:
        return self.encoder.head_size


class FlashChatGLMForConditionalGeneration(nn.Module):
    def __init__(self, config: ChatGLMConfig, weights: Weights):
        super().__init__()

        self.transformer = ChatGLMModel("transformer", config, weights)

    def forward(
        self,
        input_ids: torch.Tensor,
        position_ids: torch.Tensor,
        cu_seqlen_prefill: Optional[torch.Tensor],
        kv_cache: List[Tuple[torch.Tensor, torch.tensor]],
        block_tables: Optional[torch.Tensor],
        slots: torch.Tensor,
        input_lengths: torch.Tensor,
        max_s: int,
        lm_head_indices: Optional[torch.Tensor],
    ) -> torch.Tensor:
        hidden_states = self.transformer(
            input_ids=input_ids,
            position_ids=position_ids,
            cu_seqlen_prefill=cu_seqlen_prefill,
            kv_cache=kv_cache,
            block_tables=block_tables,
            slots=slots,
            input_lengths=input_lengths,
            max_s=max_s,
        )
        if lm_head_indices is not None:
            hidden_states = hidden_states[lm_head_indices]
        lm_logits = self.transformer.output_layer(hidden_states)
        return lm_logits

    @property
    def num_layers(self) -> int:
        return self.transformer.num_layers

    @property
    def num_kv_heads(self) -> int:
        return self.transformer.num_kv_heads

    @property
    def head_size(self) -> int:
        return self.transformer.head_size


def _resolve_prefix(name, parent: Optional[str] = None) -> str:
    if parent is None:
        return name
    return f"{parent}.{name}"


def swiglu(x: torch.Tensor) -> torch.Tensor:
    x = torch.chunk(x, 2, dim=-1)
    return F.silu(x[0]) * x[1]


def load_linear_from_packed_weights(
    config: ChatGLMConfig,
    prefix: str,
    weights: Weights,
    bias: bool,
    size_lst: List[int],
) -> TensorParallelColumnLinear:
    weight = weights.get_sharded_col_packed(
        _resolve_prefix("weight", prefix), dim=0, size_lst=size_lst
    )
    if bias:
        bias = weights.get_sharded_col_packed(
            _resolve_prefix("bias", prefix), dim=0, size_lst=size_lst
        )
    else:
        bias = None
    return TensorParallelColumnLinear(get_linear(weight, bias, config.quantize))


def load_attention(
    config: ChatGLMConfig,
    prefix: str,
    weights: Weights,
    bias: bool,
    num_heads: int,
    num_kv_heads: int,
    head_size: int,
) -> TensorParallelColumnLinear:
    weight = weights.get_sharded_col_packed_qkv(
        _resolve_prefix("weight", prefix),
        dim=0,
        num_heads=num_heads,
        num_kv_heads=num_kv_heads,
        head_size=head_size,
    )
    if bias:
        bias = weights.get_sharded_col_packed_qkv(
            _resolve_prefix("bias", prefix),
            dim=0,
            num_heads=num_heads,
            num_kv_heads=num_kv_heads,
            head_size=head_size,
        )
    else:
        bias = None
    return TensorParallelColumnLinear(get_linear(weight, bias, config.quantize))
