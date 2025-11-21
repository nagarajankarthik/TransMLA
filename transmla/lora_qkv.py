import torch
import torch.nn as nn
from typing import Optional, Tuple

from transformers.modeling_utils import ALL_ATTENTION_FUNCTIONS
from transformers.models.deepseek_v3.modeling_deepseek_v3 import apply_rotary_pos_emb_interleave

from utils import pca_calc, get_qkv_calibrate_outputs, evaluate_ppl, statistics_qkv_rmsnorm, use_original_norm_weights, use_original_norm_weights_post_proj

 
def repeat_kv(hidden_states: torch.Tensor, n_rep: int) -> torch.Tensor:
    batch, num_key_value_heads, slen, head_dim = hidden_states.shape
    if n_rep == 1:
        return hidden_states
    hidden_states = hidden_states[:, :, None, :, :].expand(
        batch, num_key_value_heads, n_rep, slen, head_dim
    )
    return hidden_states.reshape(batch, num_key_value_heads * n_rep, slen, head_dim)

class LoraQKV(nn.Module):
    def __init__(
        self, 
        self_attn, 
        query_outputs, 
        key_outputs, 
        value_outputs, 
        q_lora_rank=None,
        qk_mqa_dim=64, 
        collapse=1,
        kv_lora_rank=896,
        use_qkv_norm=False, 
        balance_kv_ratio=None, 
        rms_norm_eps=1e-6,
    ):
        super().__init__()
        assert qk_mqa_dim * collapse == self_attn.head_dim

        self.config = self_attn.config
        self.dtype = self_attn.q_proj.weight.dtype
        self.layer_idx = self_attn.layer_idx
        self.num_attention_heads = self_attn.num_attention_heads
        self.head_dim = self_attn.head_dim
        self.qk_mqa_dim = qk_mqa_dim
        self.collapse = collapse
        self.latent_dim = self_attn.latent_dim
        self.attention_dropout = self_attn.attention_dropout
        self.hidden_size = self_attn.hidden_size
        self.q_lora_rank = q_lora_rank
        self.kv_lora_rank = kv_lora_rank
        assert self.kv_lora_rank <= 2 * self.latent_dim - self.qk_mqa_dim, f"kv_lora_rank ({self.kv_lora_rank}) must be less than 2 * latent_dim ({self.latent_dim}) - qk_mqa_dim ({self.qk_mqa_dim})"

        self.attention_function = ALL_ATTENTION_FUNCTIONS["sdpa"]
        self.scaling = (self.head_dim + self.qk_mqa_dim)**(-0.5)

        # -----------------Attributes for the bias-----------------
        q_bias = self_attn.q_proj.bias is not None
        k_bias = self_attn.k_proj.bias is not None
        v_bias = self_attn.v_proj.bias is not None
        assert q_bias == k_bias == v_bias, f"q_bias ({q_bias}), k_bias ({k_bias}), v_bias ({v_bias}) must be the same"
        self.attention_bias = q_bias

        # -----------------module definitions-----------------
        # q_a_proj & q_b_proj
        if q_lora_rank is not None:
            self.q_a_proj = nn.Linear(
                self.hidden_size, 
                q_lora_rank, 
                bias=False,
                device=self_attn.q_proj.weight.device,
                dtype=self.dtype,
            )
            if use_qkv_norm:
                self.q_nope_rmsnorm = nn.RMSNorm(self.head_dim, device=self_attn.q_proj.weight.device, dtype=self.dtype, eps=rms_norm_eps)
                self.q_rope_rmsnorm = nn.RMSNorm(self.qk_mqa_dim, device=self_attn.q_proj.weight.device, dtype=self.dtype, eps=rms_norm_eps)
                # self.q_a_layernorm = nn.RMSNorm(q_lora_rank, device=self_attn.q_proj.weight.device, dtype=self.dtype, eps=rms_norm_eps)
            self.q_b_proj = nn.Linear(
                q_lora_rank,
                self.num_attention_heads * (self.qk_mqa_dim + self.head_dim), 
                bias=self.attention_bias,
                device=self_attn.q_proj.weight.device,
                dtype=self.dtype,
            )
        else:
            self.q_proj = nn.Linear(
                self.hidden_size, 
                self.num_attention_heads * (self.qk_mqa_dim + self.head_dim), 
                bias=self.attention_bias,
                device=self_attn.q_proj.weight.device,
                dtype=self.dtype,
            )
        # kv_a_proj & kv_b_proj
        self.kv_a_proj_with_mqa = nn.Linear(
            self.hidden_size,
            kv_lora_rank + qk_mqa_dim,
            bias=self.attention_bias,
            device=self_attn.k_proj.weight.device,
            dtype=self.dtype,
        )
        if use_qkv_norm:
            # self.kv_a_layernorm = nn.RMSNorm(kv_lora_rank, device=self_attn.k_proj.weight.device, dtype=self.dtype, eps=rms_norm_eps)
            self.k_nope_rmsnorm = nn.RMSNorm(self.head_dim, device=self_attn.k_proj.weight.device, dtype=self.dtype, eps=rms_norm_eps)
            self.k_rope_rmsnorm = nn.RMSNorm(self.qk_mqa_dim, device=self_attn.k_proj.weight.device, dtype=self.dtype, eps=rms_norm_eps)
        self.kv_b_proj = nn.Linear(
            kv_lora_rank,
            self.num_attention_heads * self.head_dim * 2,
            bias=False,
            device=self_attn.k_proj.weight.device,
            dtype=self.dtype,
        )
        # nothing else to do for o_proj
        self.o_proj = self_attn.o_proj

        # -----------------apply bkv on the key and value outputs-----------------
        if balance_kv_ratio is not None:
            k_outputs_norm = torch.cat([key.reshape(-1, self.latent_dim)[:,self.qk_mqa_dim:] for key in key_outputs]).norm(p=2,dim=0).mean()
            v_outputs_norm = torch.cat([value.reshape(-1, self.latent_dim)[:,self.qk_mqa_dim:] for value in value_outputs]).norm(p=2,dim=0).mean()
            ratio = k_outputs_norm / (v_outputs_norm * balance_kv_ratio)
            self_attn.k_proj.weight.data[self.qk_mqa_dim:] /= ratio
            if self.attention_bias:
                self_attn.k_proj.bias.data[self.qk_mqa_dim:] /= ratio
            self_attn.k_up_proj.weight.data[:, self.qk_mqa_dim:] *= ratio
        else:
            ratio = 1
        kv_outputs = [torch.cat([key_outputs[i][:,:,qk_mqa_dim:] / ratio, value_outputs[i]], dim=-1) for i in range(len(key_outputs))]

        # -----------------apply pca on the query and key/value outputs-----------------
        if self.q_lora_rank is not None:
            R_q = pca_calc(query_outputs, self_attn.q_proj.weight.device)
        else:
            R_q = None
        R_kv = pca_calc(kv_outputs, self_attn.k_proj.weight.device)

        # -----------------initialize the weights / bias-----------------
        self._init_weights(self_attn, R_q, R_kv)
        
    def _init_weights(self, self_attn, R_q, R_kv):
        # 0. Split the weights of k_proj and v_proj into rope / nope parts.
        k_a_rope_weight, k_a_nope_weight = self_attn.k_proj.weight.data.split([self.qk_mqa_dim, self.latent_dim - self.qk_mqa_dim],dim=0)
        k_b_rope_weight, k_b_nope_weight = self_attn.k_up_proj.weight.data.split([self.qk_mqa_dim, self.latent_dim - self.qk_mqa_dim], dim=1)
        k_b_rope_weight = k_b_rope_weight.view(self.num_attention_heads, self.head_dim, self.qk_mqa_dim)
        k_b_nope_weight = k_b_nope_weight.view(self.num_attention_heads, self.head_dim, self.latent_dim-self.qk_mqa_dim)
        
        v_a_nope_weight  = self_attn.v_proj.weight.data
        v_b_nope_weight = self_attn.v_up_proj.weight.data
        v_b_nope_weight = v_b_nope_weight.view(self.num_attention_heads, self.head_dim, self.latent_dim)

        if self.attention_bias:
            q_bias = self_attn.q_proj.bias.data
            v_bias = self_attn.v_proj.bias.data
            k_bias_rope, k_bias_nope = self_attn.k_proj.bias.data.split([self.qk_mqa_dim, self.latent_dim - self.qk_mqa_dim], dim=0)


        # 1. Initialize q_a_proj / q_b_proj if q_lora_rank is not None (revised by xiaojuan based on bias...)
        # 1.1 Initialize q_a_proj
        # Compute scaling factor to adjust attention scaling from 1/√head_dim to 1/√(head_dim + qk_mqa_dim)
        # For Qwen3-4B: original_scaling = 1/√128 ≈ 0.0884, self.scaling = 1/√192 ≈ 0.0722
        # Result: scaling ≈ 1.2247
        original_scaling = getattr(self.config, "query_pre_attn_scalar", self.head_dim)**-0.5
        scaling = original_scaling / self.scaling
        scaling = 1.0
        if self.q_lora_rank is not None:
            # Get original q_proj weight: [3,584, 4,096] for Qwen3-4B
            q_weight = self_attn.q_proj.weight.data.to(torch.float64)

            # Create q_a_proj weight via PCA projection
            # R_q shape: [4,096, 4,096] (PCA matrix from calibration)
            # R_q.T @ q_weight: [4,096, 3,584] (matrix multiplication)
            # [:q_lora_rank] takes first 512 rows: [512, 3,584]
            # Final q_a_proj.weight: [512, 3,584] (Linear(3584, 512))
            q_a_weight = (R_q.T @ q_weight)[: self.q_lora_rank].to(self.dtype)
            self.q_a_proj.weight.data = q_a_weight.contiguous()
            
            # Extract PCA basis vectors for q_b_proj
            # R_q[:, :q_lora_rank] extracts first 512 columns: [4,096, 512]
            # These are the top 512 PCA basis vectors
            q_b_weight = R_q[:, :self.q_lora_rank].to(self.dtype)
            # Reshape to per-head format: [4,096, 512] → [32, 128, 512] for Qwen3-4B
            # This reshapes: 32 heads × 128 head_dim × 512 PCA dimensions
            q_b_weight = q_b_weight.view(self.num_attention_heads, self.head_dim, self.q_lora_rank)
            # Absorb the rope part of k_b_proj into q_b_proj via einsum
            # q_b_weight: [32, 128, 512] (h=32 heads, d=128 head_dim, q=512 PCA dims)
            # k_b_rope_weight: [32, 128, 64] (from k_up_proj, h=32, d=128, k=64 rope dims)
            # einsum "hdq,hdk->hkq": contract over d (head_dim)
            # Result: [32, 64, 512] (h=32, k=64 rope dims, q=512 PCA dims)
            q_b_rope_weight = torch.einsum("hdq,hdk->hkq", q_b_weight, k_b_rope_weight)
            # Concatenate nope and rope parts, then reshape
            # q_b_weight: [32, 128, 512] (nope part)
            # q_b_rope_weight: [32, 64, 512] (rope part)
            # Concat along dim=1: [32, 192, 512] (128 + 64 = 192)
            # Reshape: [32 * 192, 512] = [6,144, 512] for Qwen3-4B
            q_b_with_mqa_weight = torch.cat([q_b_weight, q_b_rope_weight], dim=1).reshape(
                self.num_attention_heads * (self.head_dim + self.qk_mqa_dim), self.q_lora_rank
            )

            # Scale the weight before initializing the q_b_proj
            # In the original GQA, attention scores are divided by sqrt(head_dim).
            # However, in the transformed MLA, the attention scores are divided by sqrt(head_dim + qk_mqa_dim).
            # Apply scaling factor (≈1.2247) to adjust for the new attention scaling
            # Final q_b_proj.weight: [6,144, 512] (Linear(512, 6144) where 6144 = 32 heads × (128 + 64))
            self.q_b_proj.weight.data = q_b_with_mqa_weight.contiguous() * scaling

        else:
            q_weight = self_attn.q_proj.weight.data.view(self.num_attention_heads, self.head_dim, self.hidden_size)
            q_rope_weight = torch.einsum("hdD,hdk->hkD", q_weight, k_b_rope_weight) 
            q_with_mqa_weight = torch.cat([q_weight, q_rope_weight], dim=1).reshape(
                self.num_attention_heads * (self.head_dim + self.qk_mqa_dim), self.hidden_size
            )
            
            self.q_proj.weight.data = q_with_mqa_weight.contiguous() * scaling

        if self.attention_bias:
            q_bias = q_bias.reshape(self.num_attention_heads, self.head_dim)
            q_rope_bias = torch.einsum("hd,hdk->hk", q_bias.to(torch.float64), k_b_rope_weight.to(torch.float64)).to(self.dtype)
            q_bias = torch.cat([q_bias, q_rope_bias], dim=1).flatten().contiguous() * scaling
            if self.q_lora_rank is not None:
                self.q_b_proj.bias.data = q_bias
            else:
                self.q_proj.bias.data = q_bias
            
        
        # 2. Low-rank decomposing k_proj and v_proj
        # 2.1 Concatenate the nope parts of k_proj and v_proj
        # For Qwen3-4B:
        # k_a_nope_weight: [960, 3,584] (from k_proj split, latent_dim - qk_mqa_dim = 1024 - 64 = 960)
        # v_a_nope_weight: [1,024, 3,584] (v_proj.weight.data)
        # Concatenate along dim=0: [1,984, 3,584] (960 + 1024 = 1984)
        kv_a_nope_weight = torch.cat([k_a_nope_weight, v_a_nope_weight], dim=0).to(torch.float64)
        if self.attention_bias:
            # Concatenate biases: k_bias_nope [960] + v_bias [1,024] = [1,984]
            # Unsqueeze to add dimension: [1,984, 1]
            # Concatenate with weight along dim=-1: [1,984, 3,585] (3584 + 1 = 3585)
            kv_a_nope_bias = torch.cat([k_bias_nope, v_bias]).unsqueeze(-1).to(torch.float64)
            kv_a_nope_weight = torch.cat([kv_a_nope_weight, kv_a_nope_bias], dim=-1)
        # Create block-diagonal structure for kv_b_nope_weight
        # For Qwen3-4B:
        # k_b_nope_weight: [32, 128, 960] (from k_up_proj, reshaped to per-head format)
        # v_b_nope_weight: [32, 128, 1,024] (from v_up_proj, reshaped to per-head format)
        # First inner concat: [k_b_nope_weight, zeros] → [32, 128, 1,984] (960 + 1024 = 1984)
        # Second inner concat: [zeros, v_b_nope_weight] → [32, 128, 1,984]
        # Outer concat along dim=1: [64, 128, 1,984] (32 + 32 = 64 heads)
        # Reshape: [8,192, 1,984] (64 * 128 = 8192, 2 * 1024 - 64 = 1984)
        kv_b_nope_weight = torch.cat(
            [
                torch.cat([k_b_nope_weight, torch.zeros_like(v_b_nope_weight)], dim=-1),
                torch.cat([torch.zeros_like(k_b_nope_weight), v_b_nope_weight], dim=-1)
            ], 
            dim=1
        ).reshape(2 * self.num_attention_heads * self.head_dim, 2 * self.latent_dim - self.qk_mqa_dim).to(torch.float64)
        

        # 2.2 Low-rank decomposing kv_a_nope_weight and kv_b_nope_weight
        # Apply PCA to kv_a_nope_weight
        # For Qwen3-4B:
        # R_kv shape: [1,984, 1,984] (PCA matrix from calibration)
        # kv_a_nope_weight: [1,984, 3,584] (or [1,984, 3,585] if bias was included)
        # R_kv.T @ kv_a_nope_weight: [1,984, 3,584] (matrix multiplication)
        # [:kv_lora_rank] takes first 512 rows: [512, 3,584]
        # Final kv_a_nope_weight: [512, 3,584] (or [512, 3,585] if bias)
        kv_a_nope_weight = (R_kv.T @ kv_a_nope_weight)[: self.kv_lora_rank].to(self.dtype)
        if self.attention_bias:
            # Split weight and bias if bias was included in the PCA input
            # kv_a_nope_weight: [512, 3,585] (includes bias column)
            # Split: [512, 3,584] (weight) and [512, 1] (bias)
            # Flatten bias: [512]
            kv_a_nope_weight, kv_a_nope_bias = torch.split(kv_a_nope_weight, [self.hidden_size, 1], dim=-1)
            kv_a_nope_bias = kv_a_nope_bias.flatten().to(self.dtype)
        # Apply PCA to kv_b_nope_weight
        # For Qwen3-4B:
        # kv_b_nope_weight: [8,192, 1,984] (block-diagonal structure from previous step)
        # R_kv: [1,984, 1,984] (PCA matrix from calibration)
        # kv_b_nope_weight @ R_kv: [8,192, 1,984] (matrix multiplication)
        # [:, :kv_lora_rank] takes first 512 columns: [8,192, 512]
        # Final kv_b_nope_weight: [8,192, 512]
        kv_b_nope_weight = (kv_b_nope_weight @ R_kv)[:, :self.kv_lora_rank].to(self.dtype)
        # Store in kv_b_proj.weight: [8,192, 512] (Linear(512, 8192) where 8192 = 32 heads × 128 head_dim × 2)
        self.kv_b_proj.weight.data = kv_b_nope_weight.contiguous()
        # Concatenate kv_a_nope_weight with k_a_rope_weight to form kv_a_proj_with_mqa
        # For Qwen3-4B:
        # kv_a_nope_weight: [512, 3,584] (PCA output)
        # k_a_rope_weight: [64, 3,584] (from k_proj split, rope part)
        # Concatenate along dim=0: [576, 3,584] (512 + 64 = 576)
        # Final kv_a_proj_with_mqa.weight: [576, 3,584] (Linear(3584, 576))
        kv_a_proj_with_mqa_weight = torch.cat([kv_a_nope_weight, k_a_rope_weight], dim=0)
        self.kv_a_proj_with_mqa.weight.data = kv_a_proj_with_mqa_weight.contiguous()
        if self.attention_bias:
            # Concatenate biases: kv_a_nope_bias [512] + k_bias_rope [64] = [576]
            kv_a_proj_with_mqa_bias = torch.cat([kv_a_nope_bias, k_bias_rope])
            self.kv_a_proj_with_mqa.bias.data = kv_a_proj_with_mqa_bias.contiguous()


    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_value = None,
        output_attentions: bool = False,
        use_cache: bool = False,
        cache_position: Optional[torch.LongTensor] = None,
        position_embeddings: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[Tuple[torch.Tensor]]]:
        """
        Forward pass of the LoraQKV attention module implementing Multi-head Latent Attention (MLA).
        
        Processes hidden states through low-rank query/key/value projections with RMSNorm,
        applies Rotary Position Embedding (RoPE) to the RoPE parts, and computes attention.
        Assumes q_lora_rank is always used and qkv_norm is always enabled.
        
        Args:
            hidden_states: Input tensor [batch_size, seq_len, hidden_size]
            attention_mask: Optional attention mask tensor
            position_ids: Optional position IDs (deprecated)
            past_key_value: Optional cached key-value states
            output_attentions: Whether to return attention weights
            use_cache: Whether to use cached key-value states
            cache_position: Optional cache position tensor
            position_embeddings: Tuple of (cos, sin) tensors for RoPE [batch, 1, seq_len, head_dim]
        
        Returns:
            Tuple of (attn_output, attn_weights, past_key_value)
            - attn_output: [batch_size, seq_len, hidden_size]
            - attn_weights: Optional attention weights
            - past_key_value: Optional cached states
        
        Processing Steps:
        
        1. **Query Projection**:
           - q_a_proj: [B, L, H] → [B, L, q_lora_rank]
           - q_a_layernorm: Apply RMSNorm. Disabled in this version.
           - q_b_proj: [B, L, q_lora_rank] → [B, L, num_heads * (head_dim + qk_mqa_dim)]
        
        2. **Query Reshape and Split**:
           - Reshape: [B, L, num_heads * (head_dim + qk_mqa_dim)] → [B, num_heads, L, head_dim + qk_mqa_dim]
           - Split: 
            - q_nope [B, num_heads, L, head_dim], 
            - q_nope_rmsnorm: Apply RMSNorm to q_nope
            - q_rope [B, num_heads, L, qk_mqa_dim] (qk_mqa_dim = qk_rope_hidden_dim)
        
        3. **Key/Value Compression**:
           - kv_a_proj_with_mqa: [B, L, H]
           - Split: kv_nope [B, L, kv_lora_rank], k_rope [B, L, qk_mqa_dim]
           - Reshape: kv_nope [B, 1, L, kv_lora_rank], k_rope [B, 1, L, qk_mqa_dim]
        
        4. **Rotary Position Embedding**:
           - Apply RoPE to q_rope and k_rope using interleaved format with collapsed frequency
        
        5. **Query Reconstruction**:
           - Concatenate q_nope and q_rope along hidden_dim
           - Final shape: [B, num_heads, L, head_dim + qk_mqa_dim]
        
        6. **Key/Value Expansion**:
           - kv_a_layernorm: Apply RMSNorm to kv_nope [B, 1, L, kv_lora_rank]. Disabled in this version.
           - kv_b_proj: [B, 1, L, kv_lora_rank] → [B, 1, L, num_heads * head_dim * 2]
           - Reshape and transpose: [B, num_heads, L, head_dim * 2]
           - Split the hidden_dim into key nope and value parts:
             * k_nope: [B, num_heads, L, head_dim]
             * v: [B, num_heads, L, head_dim]
           - Apply RMSNorm to k_nope across head_dim
           - Duplicates k_rope for all attention heads:
             * B, 1, L, qk_mqa_dim] → [B, num_heads, L, qk_mqa_dim]
           - Concatenate k_nope and expanded k_rope along hidden_dim:
             * key_states = concat([k_nope, expanded_k_rope], dim=-1)
             * Final shape: [B, num_heads, L, head_dim + qk_mqa_dim]
        
        7. **Attention Computation**:
           - Compute scores: Q @ K^T / sqrt(head_dim + qk_mqa_dim)
           - Apply mask, softmax, and weighted sum: attention_weights @ V
        
        8. **Output Projection**:
           - Reshape: [B, num_heads, L, head_dim] → [B, L, num_heads * head_dim]
           - o_proj: [B, L, num_heads * head_dim] → [B, L, hidden_size]
        """
        bsz, q_len, _ = hidden_states.size()

        # query
        if self.q_lora_rank is not None:
            query_states = self.q_a_proj(hidden_states)
            # if hasattr(self, "q_a_layernorm"):
            #     query_states = self.q_a_layernorm(query_states)
            query_states = self.q_b_proj(query_states)
        else:
            query_states = self.q_proj(hidden_states)
        
        query_states = query_states.view(bsz, q_len, self.num_attention_heads, -1).transpose(1,2)
        q_nope, q_rope = query_states.split([self.head_dim, self.qk_mqa_dim], dim=-1)
        if hasattr(self, "q_nope_rmsnorm"):
            q_nope = self.q_nope_rmsnorm(q_nope)

        if hasattr(self, "q_rope_rmsnorm"):
            q_rope = self.q_rope_rmsnorm(q_rope)


        # key and value
        compressed_kv = self.kv_a_proj_with_mqa(hidden_states)
        kv_nope, k_rope = compressed_kv.split([self.kv_lora_rank, self.qk_mqa_dim], dim=-1)
        kv_nope = kv_nope.view(bsz, 1, q_len, self.kv_lora_rank)
        k_rope = k_rope.view(bsz, 1, q_len, self.qk_mqa_dim)

        cos, sin = position_embeddings
        q_rope, k_rope = apply_rotary_pos_emb_interleave(q_rope, k_rope, cos[ :, :, : : self.collapse], sin[ :, :, : : self.collapse])
        query_states = torch.cat([q_nope, q_rope], dim=-1)

        # Original location of kv_a_layernorm
        # if hasattr(self, "kv_a_layernorm"):
        #     kv_nope = self.kv_a_layernorm(kv_nope)
        kv_nope = self.kv_b_proj(kv_nope).view(bsz, q_len, self.num_attention_heads, self.head_dim * 2).transpose(1, 2)
        k_nope, value_states = kv_nope.split([self.head_dim, self.head_dim],dim=-1)
        if hasattr(self, "k_nope_rmsnorm"):
            k_nope = self.k_nope_rmsnorm(k_nope)
        if hasattr(self, "k_rope_rmsnorm"):
            k_rope = self.k_rope_rmsnorm(k_rope)

        key_states = torch.cat([k_nope, repeat_kv(k_rope, self.num_attention_heads)], dim=-1)

        attn_output, attn_weights = self.attention_function(
            self,
            query_states,
            key_states,
            value_states,
            attention_mask,
            dropout=0.0 if not self.training else self.attention_dropout,
            scaling=(self.head_dim)**-0.5,
            softcap=getattr(self.config, "attn_logit_softcapping", None)
        )

        attn_output = attn_output.reshape(bsz, q_len, -1).contiguous()
        attn_output = self.o_proj(attn_output)

        return attn_output, attn_weights


def low_rank_qkv(model, tokenizer, train_loader, test_loader, **kwargs):

    message = "Calibrating rope-removed model's qkv outputs"
    rm_rope_qkv_outputs = get_qkv_calibrate_outputs(model, train_loader, message)


    for layer_idx, layer in enumerate(model.model.layers):
        setattr(layer, "self_attn", LoraQKV(
            layer.self_attn,
            rm_rope_qkv_outputs["query"][layer_idx], 
            rm_rope_qkv_outputs["key"][layer_idx], 
            rm_rope_qkv_outputs["value"][layer_idx], 
            q_lora_rank=kwargs["q_lora_rank"], 
            qk_mqa_dim=kwargs["qk_mqa_dim"], 
            collapse=kwargs["collapse"],
            kv_lora_rank=kwargs["kv_lora_rank"],
            use_qkv_norm=kwargs["use_qkv_norm"],
            balance_kv_ratio=kwargs["balance_kv_ratio"],
            rms_norm_eps=model.config.rms_norm_eps,
        ))

    
    if kwargs["use_qkv_norm"]:
        if kwargs.get("use_original_norm_weights", False):
            original_norm_weights = kwargs["original_norm_weights"]
            # Use original norm weights
            for layer_idx, layer in enumerate(model.model.layers):
                norm_weights = original_norm_weights[layer_idx] if layer_idx < len(original_norm_weights) else {}
                # use_original_norm_weights(
                #     layer.self_attn,
                #     norm_weights["q_norm"],
                #     norm_weights["k_norm")
                # )
                use_original_norm_weights_post_proj(layer.self_attn, 
                                                    norm_weights["q_norm"], 
                                                    norm_weights["k_norm"])
        else:
            # Compute norm weights from calibration data
            lora_qkv_outputs = get_qkv_calibrate_outputs(model, train_loader, message="qkv check for norm weights")
            for layer_idx, layer in enumerate(model.model.layers):
                # if len(lora_qkv_outputs["q_a_proj"]) > layer_idx 
                # is used to check if q_a_proj exists for the current layer
                statistics_qkv_rmsnorm(
                    layer.self_attn, 
                    lora_qkv_outputs["q_a_proj"][layer_idx] if len(lora_qkv_outputs["q_a_proj"]) > layer_idx else None, 
                    lora_qkv_outputs["kv_a_proj"][layer_idx]
                )

    if test_loader:
        message = "Evaluating lora-qkv model's ppl"
        dataset_ppl = evaluate_ppl(model, tokenizer.pad_token_id, test_loader, message)
        print(f'Low rank approximate QKV ppl: {dataset_ppl:.4f}')
    
    return model
