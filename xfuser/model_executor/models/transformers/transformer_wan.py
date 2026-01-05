import torch
from typing import Optional

from diffusers.models.transformers.transformer_wan import WanAttnProcessor
from diffusers.models.transformers.transformer_wan import WanAttention

from xfuser.model_executor.layers.usp import USP
from xfuser.core.distributed import get_sequence_parallel_world_size
from xfuser.model_executor.layers.attention_processor import (
    set_hybrid_seq_parallel_attn,
    xFuserAttentionProcessorRegister
)
from xfuser.envs import PACKAGES_CHECKER

env_info = PACKAGES_CHECKER.get_packages_info()
HAS_LONG_CTX_ATTN = env_info["has_long_ctx_attn"]


import triton
import triton.language as tl

def get_all_config():
    configs = []
    for num_warps in [8]:
        for num_stages in [4]:
            for head_loading_stride in [8]:
                configs.append(triton.Config(
                    {"head_loading_stride":head_loading_stride,"num_stages":num_stages}, 
                    num_warps=num_warps
                ))
    return configs

@triton.autotune(
    configs=get_all_config(),
    key=['hs', 'hn', 'token_num'],
)
@triton.jit
def xdit_rope(q_ptr,k_ptr,cos_freq,sin_freq,q_out,k_out,
            hs:tl.constexpr,
            hn:tl.constexpr ,
            token_num:tl.constexpr,
            head_loading_stride:tl.constexpr,
            num_stages:tl.constexpr
            ):
    # tl.static_assert(hn % head_loading_stride == 0, "ERROR")
    program_num = tl.num_programs(0)
    for program_id in tl.range(tl.program_id(0), token_num, program_num):
        freq_offset = program_id * hs
        qk_offset = program_id * hs * hn
        half_hs:tl.constexpr = hs // 2
        num_pairs:tl.constexpr = head_loading_stride * half_hs
        pair_idx = tl.arange(0,num_pairs) # [0,1,2,...,head_loading_stride * half_hs - 1]
        head_idx_in_pair = pair_idx // half_hs # the head index for each data in each pair [0,0,0,...,0,1,1,1,...,1,...]
        pair_idx_in_head = pair_idx % half_hs # the pair index for each head [0,1,2,...,63,0,1,2,...,63]
        for head_idx in tl.range(0,hn,head_loading_stride,num_stages = num_stages):
            base_offset = qk_offset + head_idx * hs
            even_offset = base_offset + head_idx_in_pair * hs + pair_idx_in_head * 2
            odd_offset = even_offset + 1
            mask = head_idx + head_idx_in_pair < hn
            q_x1 = tl.load(q_ptr + even_offset, mask = mask)
            q_x2 = tl.load(q_ptr + odd_offset, mask = mask)
            k_x1 = tl.load(k_ptr + even_offset, mask = mask)
            k_x2 = tl.load(k_ptr + odd_offset, mask = mask)
            sin_cache = tl.load(sin_freq + freq_offset + pair_idx_in_head * 2 + 1,mask = mask)
            cos_cache = tl.load(cos_freq + freq_offset + pair_idx_in_head * 2,mask = mask)
            tl.store(q_out + even_offset,tl.cast(q_x1 * cos_cache - q_x2 * sin_cache,tl.bfloat16),mask = mask)
            tl.store(q_out + odd_offset,tl.cast(q_x1 * sin_cache + q_x2 * cos_cache,tl.bfloat16),mask = mask)
            tl.store(k_out + even_offset,tl.cast(k_x1 * cos_cache - k_x2 * sin_cache,tl.bfloat16),mask = mask)
            tl.store(k_out + odd_offset,tl.cast(k_x1 * sin_cache + k_x2 * cos_cache,tl.bfloat16),mask = mask)


@xFuserAttentionProcessorRegister.register(WanAttnProcessor)
class xFuserWanAttnProcessor(WanAttnProcessor):

    def __init__(self):
        super().__init__()
        use_long_ctx_attn_kvcache = True
        self.use_long_ctx_attn_kvcache = (
            HAS_LONG_CTX_ATTN
            and use_long_ctx_attn_kvcache
            and get_sequence_parallel_world_size() > 1
        )
        set_hybrid_seq_parallel_attn(self, self.use_long_ctx_attn_kvcache)

    def _get_qkv_projections(self, attn: "WanAttention", hidden_states: torch.Tensor, encoder_hidden_states: torch.Tensor):
        # encoder_hidden_states is only passed for cross-attention
        if encoder_hidden_states is None:
            encoder_hidden_states = hidden_states

        if attn.fused_projections:
            if attn.cross_attention_dim_head is None:
                # In self-attention layers, we can fuse the entire QKV projection into a single linear
                query, key, value = attn.to_qkv(hidden_states).chunk(3, dim=-1)
            else:
                # In cross-attention layers, we can only fuse the KV projections into a single linear
                query = attn.to_q(hidden_states)
                key, value = attn.to_kv(encoder_hidden_states).chunk(2, dim=-1)
        else:
            query = attn.to_q(hidden_states)
            key = attn.to_k(encoder_hidden_states)
            value = attn.to_v(encoder_hidden_states)
        return query, key, value

    def _get_added_kv_projections(self, attn: "WanAttention", encoder_hidden_states_img: torch.Tensor):
        if attn.fused_projections:
            key_img, value_img = attn.to_added_kv(encoder_hidden_states_img).chunk(2, dim=-1)
        else:
            key_img = attn.add_k_proj(encoder_hidden_states_img)
            value_img = attn.add_v_proj(encoder_hidden_states_img)
        return key_img, value_img


    def __call__(
        self,
        attn: "WanAttention",
        hidden_states: torch.Tensor,
        encoder_hidden_states: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        rotary_emb: Optional[tuple[torch.Tensor, torch.Tensor]] = None,
    ) -> torch.Tensor:
        encoder_hidden_states_img = None
        if attn.add_k_proj is not None:
            # 512 is the context length of the text encoder, hardcoded for now
            image_context_length = encoder_hidden_states.shape[1] - 512
            encoder_hidden_states_img = encoder_hidden_states[:, :image_context_length]
            encoder_hidden_states = encoder_hidden_states[:, image_context_length:]

        query, key, value = self._get_qkv_projections(attn, hidden_states, encoder_hidden_states)

        query = attn.norm_q(query)
        key = attn.norm_k(key)

        query = query.unflatten(2, (attn.heads, -1))
        key = key.unflatten(2, (attn.heads, -1))
        value = value.unflatten(2, (attn.heads, -1))

        if rotary_emb is not None:
            bs,seq_q,hn_q,hs = query.shape
            _,seq_k,hn_k,_ = key.shape
            if seq_q == seq_k and hn_q == hn_k:
                xdit_rope[(seq_q,1,1)](query, key, rotary_emb[0],rotary_emb[1], query, key, hs, hn_q, seq_q)
            else:
                def apply_rotary_emb(
                    hidden_states: torch.Tensor,
                    freqs_cos: torch.Tensor,
                    freqs_sin: torch.Tensor,
                ):
                    x1, x2 = hidden_states.unflatten(-1, (-1, 2)).unbind(-1)
                    cos = freqs_cos[..., 0::2]
                    sin = freqs_sin[..., 1::2]
                    out = torch.empty_like(hidden_states)
                    out[..., 0::2] = x1 * cos - x2 * sin
                    out[..., 1::2] = x1 * sin + x2 * cos
                    return out.type_as(hidden_states)

                query = apply_rotary_emb(query, *rotary_emb)
                key = apply_rotary_emb(key, *rotary_emb)

        # I2V task
        hidden_states_img = None
        if encoder_hidden_states_img is not None:
            key_img, value_img = self._get_added_kv_projections(attn, encoder_hidden_states_img)
            key_img = attn.norm_added_k(key_img)

            key_img = key_img.unflatten(2, (attn.heads, -1))
            value_img = value_img.unflatten(2, (attn.heads, -1))


            hidden_states_img = USP(query.transpose(1, 2), key_img.transpose(1, 2), value_img.transpose(1, 2)).transpose(1, 2)
            hidden_states_img = hidden_states_img.flatten(2, 3)
            hidden_states_img = hidden_states_img.type_as(query)

        hidden_states = USP(query.transpose(1, 2), key.transpose(1, 2), value.transpose(1, 2)).transpose(1, 2)
        hidden_states = hidden_states.flatten(2, 3)
        hidden_states = hidden_states.type_as(query)

        if hidden_states_img is not None:
            hidden_states = hidden_states + hidden_states_img

        hidden_states = attn.to_out[0](hidden_states)
        hidden_states = attn.to_out[1](hidden_states)
        return hidden_states
