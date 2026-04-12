from functools import partial
from torch import Tensor
import torch.nn as nn
import torch
import torch.nn.functional as F
from einops import einsum, rearrange, repeat
from einops.layers.torch import Rearrange
import einx
from rotary_embedding_torch import RotaryEmbedding

from .model_utils import *


LinearNoBias = partial(nn.Linear, bias=False)


class PreLayerNorm(nn.Module):
    def __init__(
        self,
        fn,
        *,
        dim
    ):
        super().__init__()
        self.fn = fn
        self.norm = nn.LayerNorm(dim)
        
    def forward(
        self, 
        x, 
        **kwargs
    ):  # (..., n, d) -> (..., n, d)
        x = self.norm(x)
        return self.fn(x, **kwargs)
    

# row/col-wise dropout
class Dropout(nn.Module):
    def __init__(
        self,
        prob: float,
        *,
        dropout_type = None, # ['row', 'col'] | None
    ):
        super().__init__()
        self.dropout = nn.Dropout(prob)
        self.dropout_type = dropout_type
        
    def forward(
        self,
        t: Tensor
    ):
        if self.dropout_type in {'row', 'col'}:
            assert t.ndim == 4, 'tensor must be 4 dimensions for row / col structured dropout'
        
        if not exists(self.dropout_type):
            return self.dropout(t)
        
        if self.dropout_type == 'row':
            batch, _, col, dim = t.shape
            ones_shape = (batch, 1, col, dim)

        elif self.dropout_type == 'col':
            batch, row, _, dim = t.shape
            ones_shape = (batch, row, 1, dim)
        
        ones = t.new_ones(ones_shape)
        dropped = self.dropout(ones)
        return t * dropped
    

class SwiGLU(nn.Module):
    def forward(
        self,
        x, # (..., d)
    ): # (..., d//2)

        x, gates = x.chunk(2, dim=-1)
        return F.silu(gates) * x
    

# multi-head attention
class Attention(nn.Module):
    def __init__(
        self,
        *,
        dim,
        dim_head=64,
        heads=8,
        dropout=0.,
        gate_output=True, # False for LLMs，True for PairFormer
        query_bias=True,
        use_rotary_embedding=False,
        num_memory_kv: int = 0,
        enable_attn_softclamp=False,
        attn_softclamp_value=50.,
        softmax_full_precision=False
    ):
        super().__init__()
        """
        ein notation:

        b - batch
        h - heads
        n - sequence
        d - dimension
        e - dimension (pairwise rep)
        i - source sequence
        j - context sequence
        m - memory key / value seq
        """

        dim_inner = dim_head * heads

        self.attend = Attend(
            dropout=dropout,
            enable_attn_softclamp=enable_attn_softclamp,
            attn_softclamp_value=attn_softclamp_value,
            softmax_full_precision=softmax_full_precision
        )

        self.split_heads = Rearrange('b n (h d) -> b h n d', h=heads)
        self.merge_heads = Rearrange('b h n d -> b n (h d)')

        self.to_q = nn.Linear(dim, dim_inner, bias=query_bias)
        self.to_kv = LinearNoBias(dim, dim_inner * 2)
        self.to_out = LinearNoBias(dim_inner, dim)

        self.use_rotary_embedding = use_rotary_embedding
        if self.use_rotary_embedding:
            self.rotary_emb = RotaryEmbedding(dim = dim_head)

        self.memory_kv = None
        if num_memory_kv > 0:
            self.memory_kv = nn.Parameter(
                torch.zeros(2, heads, num_memory_kv, dim_head))
            nn.init.normal_(self.memory_kv, std=0.02)

        self.to_gates = None
        if gate_output:
            self.to_gates = nn.Sequential(LinearNoBias(dim, dim_inner), nn.Sigmoid())

    def forward(
        self,
        seq, # (b, i, d)
        mask=None,  # (b, n) | None
        context=None, # (b, j, d) | None
        attn_bias=None # (b, i, j) | None
    ): # -> (b, i, d)

        q = self.to_q(seq)

        context_seq = default(context, seq)
        k, v = self.to_kv(context_seq).chunk(2, dim=-1)

        q, k, v = tuple(self.split_heads(t) for t in (q, k, v))
        
        # add rotary embedding here
        if self.use_rotary_embedding:
            q = self.rotary_emb.rotate_queries_or_keys(q)
            k = self.rotary_emb.rotate_queries_or_keys(k)

        # attention
        out = self.attend(
            q, k, v,
            attn_bias=attn_bias,
            mask=mask,
            memory_kv=self.memory_kv
        )

        # merge heads
        out = self.merge_heads(out)

        # gate output
        if exists(self.to_gates):
            gates = self.to_gates(seq)
            out = out * gates

        # combine heads
        return self.to_out(out)


class Attend(nn.Module):
    def __init__(
        self,
        dropout=0.,
        scale: float | None = None,
        enable_attn_softclamp=False,
        attn_softclamp_value=50.,
        softmax_full_precision=False
    ):
        super().__init__()
        """
        ein notation:

        b - batch
        h - heads
        n - sequence
        d - dimension
        e - dimension (pairwise rep)
        i - source sequence
        j - context sequence
        """

        self.scale = scale
        self.dropout = dropout

        self.attn_dropout = nn.Dropout(dropout)

        # softclamp attention logits
        # being adopted by a number of recent llms (gemma, grok)
        self.enable_attn_softclamp = enable_attn_softclamp
        self.attn_softclamp_value = attn_softclamp_value

        # whether to use full precision for softmax
        self.softmax_full_precision = softmax_full_precision

    def forward(
        self,
        q, # (b, h, i, d)
        k, # (b, h, j, d)
        v, # (b, h, j, d)
        mask=None, # (b, j) | None
        attn_bias=None, # (..., i, j) | None
        memory_kv=None  # (2, h, m, d) | None
    ): # -> (b, i, j, d)

        dtype = q.dtype

        if exists(memory_kv):
            batch, num_mem_kv = q.shape[0], memory_kv.shape[-2]

            mk, mv = memory_kv
            mk, mv = tuple(repeat(t, 'h m d -> b h m d', b=batch)
                           for t in (mk, mv))
            k = torch.cat((mk, k), dim=-2)
            v = torch.cat((mv, v), dim=-2)

            if exists(attn_bias):
                attn_bias = pad_at_dim(attn_bias, (num_mem_kv, 0), value=0.)

            if exists(mask):
                mask = pad_at_dim(mask, (num_mem_kv, 0), value=True)

        # default attention
        scale = default(self.scale, q.shape[-1] ** -0.5)
        q = q * scale

        # similarity
        sim = einsum(q, k, "b h i d, b h j d -> b h i j")

        # attn bias
        if exists(attn_bias):
            sim = sim + attn_bias

        # maybe softclamp
        if self.enable_attn_softclamp:
            sim = softclamp(sim, self.attn_softclamp_value)

        # masking
        if exists(mask):
            sim = einx.where(
                'b j, b h i j, -> b h i j',
                mask.bool(), sim, max_neg_value(sim)
            )

        # attention cast float32 - in case there are instabilities with float16
        softmax_kwargs = dict()

        if self.softmax_full_precision:
            softmax_kwargs.update(dtype=torch.float32)

        # attention
        attn = sim.softmax(dim=-1, **softmax_kwargs)
        attn = attn.to(dtype)

        attn = self.attn_dropout(attn)

        # aggregate values
        out = einsum(attn, v, "b h i j, b h j d -> b h i d")

        return out
    

class LMHead(nn.Module):

    def __init__(self, embed_dim, output_dim, weight):
        super().__init__()
        self.dense = nn.Linear(embed_dim, embed_dim)
        self.layer_norm = nn.LayerNorm(embed_dim)
        self.weight = weight
        self.bias = nn.Parameter(torch.zeros(output_dim))

    def forward(self, features):
        x = self.dense(features)
        x = F.gelu(x)
        x = self.layer_norm(x)
        # project back to size of vocabulary with bias
        x = F.linear(x, self.weight) + self.bias
        return x


class TransformerLayer(nn.Module):
    def __init__(
        self,
        dim,
        dim_head=64,
        heads=4,
        ffn_dim=512,
        dropout=0.1,
        gate_output=False,
        use_rotary_embedding=False, # False
        **kwargs):
        super().__init__()
        
        # Attention block with PreLayerNorm
        self.attn = PreLayerNorm(
            Attention(
                dim=dim,
                dim_head=dim_head,
                heads=heads,
                dropout=dropout,
                gate_output=gate_output,
                use_rotary_embedding=use_rotary_embedding,
                **kwargs
            ),
            dim=dim
        )
        
        # Feed-forward block with PreLayerNorm
        self.ffn = PreLayerNorm(
            nn.Sequential(
                nn.Linear(dim, ffn_dim * 2),
                SwiGLU(), # replace gelu activation with swiglu
                nn.Dropout(dropout),
                nn.Linear(ffn_dim, dim),
                nn.Dropout(dropout)
            ),
            dim=dim
        )

    def forward(self, x, mask=None, context=None, attn_bias=None):
        # Apply attention with pre-layer normalization
        x = x + self.attn(x, mask=mask, context=context, attn_bias=attn_bias)
        # Apply feed-forward network with pre-layer normalization
        x = x + self.ffn(x)
        
        return x
    

class TransformerLayerWithAdaLN(nn.Module):
    def __init__(
        self,
        dim,
        dim_head=64,
        heads=4,
        ffn_dim=512,
        dropout=0.1,
        gate_output=False,
        use_rotary_embedding=False, # False
        **kwargs):
        super().__init__()
        
        # Attention block
        self.attn = Attention(
            dim=dim,
            dim_head=dim_head,
            heads=heads,
            dropout=dropout,
            gate_output=gate_output,
            use_rotary_embedding=use_rotary_embedding,
            **kwargs
        )
        self.norm1 = nn.LayerNorm(dim, elementwise_affine=False, eps=1e-6)
        
        # Feed-forward block
        self.ffn = nn.Sequential(
            nn.Linear(dim, ffn_dim * 2),
            SwiGLU(), # replace gelu activation with swiglu
            nn.Dropout(dropout),
            nn.Linear(ffn_dim, dim),
            nn.Dropout(dropout)
        )
        self.norm2 = nn.LayerNorm(dim, elementwise_affine=False, eps=1e-6)
        
        # adaLN modulation
        self.adaLN_modulation = nn.Sequential(
            nn.SiLU(),
            nn.Linear(dim, 6 * dim, bias=True)
        )
        self.initialize_weights()
        
    def initialize_weights(self):
        # Zero-out adaLN modulation layers in DiT blocks
        nn.init.constant_(self.adaLN_modulation[-1].weight, 0)
        nn.init.constant_(self.adaLN_modulation[-1].bias, 0)
    
    @staticmethod
    def modulate(x, shift, scale):
        return x * (1 + scale.unsqueeze(1)) + shift.unsqueeze(1)
    
    def forward(self, x, cond_feat, mask=None, context=None, attn_bias=None):
        # cond_feat can be cdr12 features or pmhc binding features
        
        shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = self.adaLN_modulation(cond_feat).chunk(6, dim=1)
        
        # Apply attention with adaptive layernorm conditioning on cdr12 features
        x = x + gate_msa.unsqueeze(1) * self.attn(self.modulate(self.norm1(x), shift_msa, scale_msa),
                                                  mask=mask, context=context, attn_bias=attn_bias)
        
        # Apply feed-forward network with adaptive layernorm conditioning on cdr12 features
        x = x + gate_mlp.unsqueeze(1) * self.ffn(self.modulate(self.norm2(x), shift_mlp, scale_mlp))
        
        return x
    
    
class ConvEncoder(nn.Module):
    def __init__(self, in_dim, hid_dim, max_seq_len):
        super().__init__()
        self.embed_layer = nn.Linear(in_dim, hid_dim)
        self.encoder = nn.Sequential(
            nn.Conv1d(hid_dim, hid_dim, kernel_size=3, padding=1),
            nn.BatchNorm1d(hid_dim),
            nn.ReLU(),
            nn.Conv1d(hid_dim, hid_dim, kernel_size=3, padding=1),
            nn.BatchNorm1d(hid_dim),
            nn.ReLU(),
        )
        self.seq2vec = nn.Sequential(
            nn.Flatten(),
            nn.Linear(max_seq_len * hid_dim, hid_dim),
            nn.ReLU()
        )
        
    def forward(self, x):
        embed = self.embed_layer(x)
        embed = self.encoder(embed.transpose(1, 2)) # b, n, d => b, d, n
        
        vec = self.seq2vec(embed)
        return vec
       
       
# Try removing triangle attention/update and see if it works
class PairwiseBlockSimple(nn.Module):
    def __init__(
        self,
        *,
        dim_pairwise = 128,
    ):
        super().__init__()
        
        pre_ln = partial(PreLayerNorm, dim = dim_pairwise)
        # transition
        self.pairwise_transition = pre_ln(Transition(dim=dim_pairwise))
        
    def forward(
        self,
        *,
        pairwise_repr, # (b, n, n, d)
        mask=None, # (b, n) | None
    ):
          
        pairwise_repr = self.pairwise_transition(pairwise_repr) + pairwise_repr
        return pairwise_repr


# consists of all the "Triangle" modules + Transition
class PairwiseBlock(nn.Module):
    def __init__(
        self,
        *,
        dim_pairwise = 128,
        tri_mult_dim_hidden = None,
        tri_attn_dim_head = 32,
        tri_attn_heads = 4,
        dropout_row_prob = 0.25,
        dropout_col_prob = 0.25,
    ):
        super().__init__()
        
        pre_ln = partial(PreLayerNorm, dim = dim_pairwise)
        
        tri_mult_kwargs = dict(
            dim=dim_pairwise,
            dim_hidden=tri_mult_dim_hidden
        )

        tri_attn_kwargs = dict(
            dim=dim_pairwise,
            heads=tri_attn_heads,
            dim_head=tri_attn_dim_head
        )

        # triangle update using outgoing edges
        self.tri_mult_outgoing = pre_ln(TriangleMultiplication(
            mix='outgoing', dropout=dropout_row_prob, dropout_type='row', **tri_mult_kwargs))
        # triangle update using incoming edges
        self.tri_mult_incoming = pre_ln(TriangleMultiplication(
            mix='incoming', dropout=dropout_row_prob, dropout_type='row', **tri_mult_kwargs))
        # triangle self-attention around starting node
        self.tri_attn_starting = pre_ln(TriangleAttention(
            node_type='starting', dropout=dropout_row_prob, dropout_type='row', **tri_attn_kwargs))
        # triangle self-attention around ending node
        self.tri_attn_ending = pre_ln(TriangleAttention(
            node_type='ending', dropout=dropout_col_prob, dropout_type='col', **tri_attn_kwargs))
        # transition
        self.pairwise_transition = pre_ln(Transition(dim=dim_pairwise))
        
    def forward(
        self,
        *,
        pairwise_repr, # (b, n, n, d)
        mask, # (b, n) | None
    ):
        # pairwise_repr = self.tri_mult_outgoing(
        #     pairwise_repr, mask=mask) + pairwise_repr
        # pairwise_repr = self.tri_mult_incoming(
        #     pairwise_repr, mask=mask) + pairwise_repr
        # pairwise_repr = self.tri_attn_starting(
        #     pairwise_repr, mask=mask) + pairwise_repr
        # pairwise_repr = self.tri_attn_ending(
        #     pairwise_repr, mask=mask) + pairwise_repr

        # pairwise_repr = self.pairwise_transition(pairwise_repr) + pairwise_repr
        # return pairwise_repr
    
        # Process outgoing and incoming in parallel if they don't depend on each other
        outgoing = self.tri_mult_outgoing(pairwise_repr, mask=mask)
        incoming = self.tri_mult_incoming(pairwise_repr, mask=mask)
        pairwise_repr = pairwise_repr + outgoing + incoming # accelarting
        
        # Similarly for attention operations if possible
        attn_start = self.tri_attn_starting(pairwise_repr, mask=mask)
        attn_end = self.tri_attn_ending(pairwise_repr, mask=mask)
        pairwise_repr = pairwise_repr + attn_start + attn_end
        
        pairwise_repr = self.pairwise_transition(pairwise_repr) + pairwise_repr
        return pairwise_repr


# triangle multiplication moduel
class TriangleMultiplication(nn.Module):
    def __init__(
        self,
        *,
        dim,
        dim_hidden = None,
        mix='incoming', # ['incoming', 'outgoing']
        dropout=0.0,
        dropout_type=None # ['row', 'col'] | None
    ):
        super().__init__()
        
        dim_hidden = default(dim_hidden, dim)
        
        self.left_right_proj = nn.Sequential(
            LinearNoBias(dim, dim_hidden * 4),
            nn.GLU(dim=-1)
        )
        
        self.out_gate = LinearNoBias(dim, dim_hidden)
        
        if mix == 'outgoing':
            self.mix_einsum_eq = '... i k d, ... j k d -> ... i j d'
        elif mix == 'incoming':
            self.mix_einsum_eq = '... k j d, ... k i d -> ... i j d'
        
        self.to_out_norm = nn.LayerNorm(dim_hidden)

        self.to_out = nn.Sequential(
            LinearNoBias(dim_hidden, dim),
            Dropout(dropout, dropout_type=dropout_type)
        )
    
    def forward(
        self,
        x, # (b, n, n, d)
        mask = None # (b, n) | None
    ):
        if exists(mask):
            mask = to_pairwise_mask(mask)
            mask = rearrange(mask, '... -> ... 1')
            
        left, right = self.left_right_proj(x).chunk(2, dim=-1) # seems too simple (maybe refer to TCRfinder)
        
        if exists(mask):
            left = left * mask
            right = right * mask
            
        out = einsum(left, right, self.mix_einsum_eq)

        out = self.to_out_norm(out)

        out_gate = self.out_gate(x).sigmoid()

        return self.to_out(out) * out_gate


# triangle self-attention module
class TriangleAttention(nn.Module):
    def __init__(
        self,
        *,
        dim,
        heads,
        node_type, # ['starting', 'ending']
        dropout=0.0,
        dropout_type=None, # ['row', 'col'] | None
        **attn_kwargs
    ):
        super().__init__()
        
        self.need_transpose = node_type == 'ending'
        
        self.attn = Attention(dim=dim, heads=heads, use_rotary_embedding=False, **attn_kwargs) # if not use rotary embedding, set use_rotary_embedding=False

        self.dropout = Dropout(dropout, dropout_type=dropout_type)

        self.to_attn_bias = nn.Sequential(
            LinearNoBias(dim, heads),
            Rearrange('... i j h -> ... h i j')
        )

    def forward(
        self,
        pairwise_repr, # (b, n, n, d)
        mask=None, # (b, n) | None
        **kwargs
    ): # -> (b, n, n, d)

        if self.need_transpose:
            pairwise_repr = rearrange(pairwise_repr, 'b i j d -> b j i d')

        attn_bias = self.to_attn_bias(pairwise_repr)

        batch_repeat = pairwise_repr.shape[1]
        attn_bias = repeat(attn_bias, 'b ... -> (b repeat) ...', repeat=batch_repeat)

        if exists(mask):
            mask = repeat(mask, 'b ... -> (b repeat) ...', repeat=batch_repeat)

        pairwise_repr, unpack_one = pack_one(pairwise_repr, '* n d')

        out = self.attn(
            pairwise_repr,
            mask=mask,
            attn_bias=attn_bias,
            **kwargs
        )

        out = unpack_one(out)

        if self.need_transpose:
            out = rearrange(out, 'b j i d -> b i j d')

        return self.dropout(out)
    

# transition module
class Transition(nn.Module):
    def __init__(
        self,
        *,
        dim,
        expansion_factor=2
    ):
        super().__init__()
        dim_inner = int(dim * expansion_factor)

        self.ff = nn.Sequential(
            LinearNoBias(dim, dim_inner * 2),
            SwiGLU(),
            LinearNoBias(dim_inner, dim)
        )

    def forward(
        self,
        x # (..., d)
    ): # (..., d)

        return self.ff(x)
    

class AttentionPairBias(nn.Module):
    """An Attention module with pair bias computation."""
    def __init__(self, *, heads, dim_pairwise, num_memory_kv=0, **attn_kwargs):
        super().__init__()

        self.attn = Attention(
            heads=heads, num_memory_kv=num_memory_kv, use_rotary_embedding=False, **attn_kwargs # if not use rotary embedding, set use_rotary_embedding=False
        )

        self.to_attn_bias_norm = nn.LayerNorm(dim_pairwise)
        self.to_attn_bias = nn.Sequential(LinearNoBias(
            dim_pairwise, heads), Rearrange("b ... h -> b h ..."))

    def forward(
        self,
        single_repr, # (b, n, ds)
        *,
        pairwise_repr, # (b, n, n, dp)
        attn_bias=None, # (b, n, n)
        **kwargs,
    ):  # (b, n, ds)
        """Perform the forward pass.

        :param single_repr: The single representation tensor.
        :param pairwise_repr: The pairwise representation tensor.
        :param attn_bias: The attention bias tensor.
        :return: The output tensor.
        """
        b, dp = pairwise_repr.shape[0], pairwise_repr.shape[-1]
        dtype, device = pairwise_repr.dtype, pairwise_repr.device

        # attention bias preparation with further addition from pairwise repr
        if exists(attn_bias):
            attn_bias = rearrange(attn_bias, "b ... -> b 1 ...")
        else:
            attn_bias = 0.0

        if pairwise_repr.numel() > float("inf"):
            # create a stub tensor and normalize it to maintain gradients to `to_attn_bias_norm`
            stub_pairwise_repr = torch.zeros((b, dp), dtype=dtype, device=device)
            stub_attn_bias_norm = self.to_attn_bias_norm(stub_pairwise_repr) * 0.0

            # adjust `attn_bias_norm` dimensions to match `pairwise_repr`
            attn_bias_norm = pairwise_repr + stub_attn_bias_norm[:, None, None, :]

            # apply bias transformation
            attn_bias = self.to_attn_bias(attn_bias_norm) + attn_bias
        else:
            attn_bias = self.to_attn_bias(self.to_attn_bias_norm(pairwise_repr)) + attn_bias

        out = self.attn(single_repr, attn_bias=attn_bias, **kwargs)

        return out


class ResidueConvBlock(nn.Module):
    def __init__(self, embed_dim, kernel_size, padding, dropout=0.0):
        super().__init__()
        self.layer = nn.Sequential(
            nn.Conv1d(embed_dim, embed_dim, kernel_size, stride=1, padding=padding),
            nn.BatchNorm1d(embed_dim),
            nn.ReLU(),
            nn.Dropout(p=dropout)
        )

    def forward(self, x):
        residual = x
        x = self.layer(x)
        x = residual + x

        return x


class ResidueConvBlock2D(nn.Module):
    def __init__(self, embed_dim, kernel_size, padding, dropout=0.0):
        super().__init__()
        self.layer = nn.Sequential(
            nn.Conv2d(embed_dim, embed_dim, kernel_size, stride=1, padding=padding),
            nn.BatchNorm2d(embed_dim),
            nn.ReLU(),
            nn.Dropout(p=dropout)
        )

    def forward(self, x):
        residual = x
        x = self.layer(x)
        x = residual + x

        return x


class MHCConvLayers(nn.Module):
    
    def __init__(self, num_layers, in_dim, embed_dim, mhc_len, kernel_size, dropout):
        super().__init__()
        self.num_layers = num_layers
        self.in_dim = in_dim
        self.embed_dim = embed_dim
        self.mhc_len = mhc_len
        self.kernel_size = kernel_size
        self.padding = (kernel_size - 1) // 2
        self.dropout = dropout
        
        self.conv = nn.Conv1d(self.in_dim, self.embed_dim, 1, 1, bias=False)
        self.layers = nn.ModuleList(
            [
                ResidueConvBlock(self.embed_dim, self.kernel_size, self.padding, self.dropout) for _ in range(self.num_layers - 1)
            ]
        )
        self.bn = nn.BatchNorm1d(self.embed_dim)

        # add positional information of MHC
        self.positional_embedding = nn.Parameter(torch.zeros(1, embed_dim, mhc_len))
        nn.init.normal_(self.positional_embedding, std=0.02)
        
    def forward(self, x):  # x: b, n, d
        x = x.transpose(1, 2)  # b, n, d => b, d, n
        x = self.conv(x)
        x = self.bn(x)
        
        # Add positional embeddings
        x = x + self.positional_embedding[:, :, :x.size(2)]

        for layer in self.layers:
            x = layer(x)
            
        return x.transpose(1, 2) # b, d, n => b, n, d


class PairFormerStack(nn.Module):
    
    def __init__(
        self,
        *,
        dim_single = 384,
        dim_pairwise = 128,
        depth = 48,
        recurrent_depth = 1,
        pair_bias_attn_dim_head = 64,
        pair_bias_attn_heads = 16,
        dropout_row_prob = 0.25,
        num_register_tokens = 0,
        pairwise_block_kwargs: dict = dict(),
        pair_bias_attn_kwargs: dict = dict()
    ):
        super().__init__()
        layers = nn.ModuleList([])
        
        pair_bias_attn_kwargs = dict(
            dim = dim_single,
            dim_pairwise = dim_pairwise,
            heads = pair_bias_attn_heads,
            dim_head = pair_bias_attn_dim_head,
            dropout = dropout_row_prob,
            **pair_bias_attn_kwargs
        )
        
        # number of PairFormer Blocks
        for _ in range(depth):
            single_pre_ln = partial(PreLayerNorm, dim = dim_single)
            
            pairwise_block = PairwiseBlock(
                dim_pairwise = dim_pairwise,
                **pairwise_block_kwargs
            )
            # remove triangle attention and update layers
            # pairwise_block = PairwiseBlockSimple(
            #     dim_pairwise = dim_pairwise,
            # )
            pair_bias_attn = AttentionPairBias(**pair_bias_attn_kwargs)
            single_transition = Transition(dim=dim_single)

            layers.append(nn.ModuleList([
                pairwise_block,
                single_pre_ln(pair_bias_attn),
                single_pre_ln(single_transition),
            ]))
            
        self.layers = layers
        
        assert recurrent_depth > 0
        self.recurrent_depth = recurrent_depth

        self.num_registers = num_register_tokens
        self.has_registers = num_register_tokens > 0

        if self.has_registers:
            self.single_registers = nn.Parameter(torch.zeros(num_register_tokens, dim_single))
            self.pairwise_row_registers = nn.Parameter(torch.zeros(num_register_tokens, dim_pairwise))
            self.pairwise_col_registers = nn.Parameter(torch.zeros(num_register_tokens, dim_pairwise))
        
    def to_layers(
        self,
        *,
        single_repr, # (b, n, ds)
        pairwise_repr, # (b, n, n, dp)
        mask=None # (b, n) | None
    ): # -> Tuple[(b, n, ds), (b, n, dp)]
        
        for _ in range(self.recurrent_depth):
            for (
                pairwise_block,
                pair_bias_attn,
                single_transition
            ) in self.layers:
                
                pairwise_repr = pairwise_block(pairwise_repr = pairwise_repr, mask = mask)
                
                single_repr = pair_bias_attn(single_repr, pairwise_repr = pairwise_repr, mask = mask) + single_repr
                single_repr = single_transition(single_repr) + single_repr
                
        return single_repr, pairwise_repr
    
    def forward(
        self,
        *,
        single_repr, # (b, n, ds)
        pairwise_repr, # (b, n, n, dp)
        mask=None # (b, n) | None
    ): # -> Tuple[(b, n, ds), (b, n, n, dp)]
        
        # prepend register tokens
        if self.has_registers:
            batch_size, num_registers = single_repr.shape[0], self.num_registers
            single_registers = repeat(self.single_registers, 'r d -> b r d', b = batch_size)
            single_repr = torch.cat((single_registers, single_repr), dim = 1)
            
            row_registers = repeat(self.pairwise_row_registers, 'r d -> b r n d', b = batch_size, n = pairwise_repr.shape[-2])
            pairwise_repr = torch.cat((row_registers, pairwise_repr), dim = 1)
            col_registers = repeat(self.pairwise_col_registers, 'r d -> b n r d', b = batch_size, n = pairwise_repr.shape[1])
            pairwise_repr = torch.cat((col_registers, pairwise_repr), dim = 2)
            
            if exists(mask):
                mask = F.pad(mask, (num_registers, 0), value = True)
        
        to_layer_fn = self.to_layers
        
        # main transfomer block layers
        single_repr, pairwise_repr = to_layer_fn(
            single_repr = single_repr,
            pairwise_repr = pairwise_repr,
            mask = mask
        )
        
        # splice out registers
        if self.has_registers:
            single_repr = single_repr[:, num_registers:]
            pairwise_repr = pairwise_repr[:, num_registers:, num_registers:]
            
        return single_repr, pairwise_repr
    

class LearnedPositionalEmbedding(nn.Embedding):
    """
    This module learns positional embeddings up to a fixed maximum size.
    Padding ids are ignored by either offsetting based on padding_idx
    or by setting padding_idx to None and ensuring that the appropriate
    position ids are passed to the forward function.
    """

    def __init__(self, num_embeddings: int, embedding_dim: int, padding_idx: int):
        if padding_idx is not None:
            num_embeddings_ = num_embeddings + padding_idx + 1
        else:
            num_embeddings_ = num_embeddings
        super().__init__(num_embeddings_, embedding_dim, padding_idx)
        self.max_positions = num_embeddings

    def forward(self, input: torch.Tensor):
        """Input is expected to be of size [bsz x seqlen]."""
        if input.size(1) > self.max_positions:
            raise ValueError(
                f"Sequence length {input.size(1)} above maximum "
                f" sequence length of {self.max_positions}"
            )
        mask = input.ne(self.padding_idx).int()
        positions = (torch.cumsum(mask, dim=1).type_as(mask) * mask).long() + self.padding_idx
        
        return F.embedding(
            positions,
            self.weight,
            self.padding_idx,
            self.max_norm,
            self.norm_type,
            self.scale_grad_by_freq,
            self.sparse,
        )

       
class LearnedChainPositionalEmbedding(nn.Module):

    def __init__(self, num_embeddings: int, embedding_dim: int, selected_chain_idx: int):
        super(LearnedChainPositionalEmbedding, self).__init__()
        self.max_positions = num_embeddings
        self.embedding_dim = embedding_dim
        self.selected_chain_idx = selected_chain_idx
        
        self.embedding_layer = nn.Embedding(self.max_positions, self.embedding_dim)

    def forward(self, chain_mask: torch.Tensor):
        mask = chain_mask.eq(self.selected_chain_idx).int()
        positions = (torch.cumsum(mask, dim=1).type_as(mask) * mask).long()
        
        return self.embedding_layer(positions)


class PairwiseEmbedding(nn.Module):
    def __init__(self, in_dim, out_dim, method = 'attention'): # method: ['attention', 'add']
        super().__init__()
        self.method = method
        if self.method == 'attention':
            self.pairwise_linear = LinearNoBias(in_dim, out_dim)
        elif self.method == 'add':
            self.linear_1 = nn.Linear(in_dim, out_dim)
            self.linear_2 = nn.Linear(in_dim, out_dim)
            
        elif self.method == 'concat':
            self.linear_1 = nn.Linear(in_dim, out_dim // 2)
            self.linear_2 = nn.Linear(in_dim, out_dim // 2)

    def forward(self, repr_1, repr_2):
        if self.method == 'attention':
            attn = torch.bmm(repr_1, repr_2.transpose(1, 2))  
            pairwise_repr = torch.einsum('bij,bik->bijk', attn, repr_1) 
            return self.pairwise_linear(pairwise_repr)
        
        elif self.method == 'add':
            return self.linear_1(repr_1).unsqueeze(2) + self.linear_2(repr_2).unsqueeze(1)
        
        elif self.method == 'concat':            
            out1 = torch.tile(self.linear_1(repr_1).unsqueeze(2), (1, 1, repr_2.shape[1], 1))
            out2 = torch.tile(self.linear_2(repr_2).unsqueeze(1), (1, repr_1.shape[1], 1, 1))
            
            return torch.cat([out1, out2], dim=-1)

class PairwiseCNN(nn.Module):
    def __init__(self, num_layers, in_dim, embed_dim, kernel_size, dropout):
        super().__init__()
        self.num_layers = num_layers
        self.in_dim = in_dim
        self.embed_dim = embed_dim
        self.kernel_size = kernel_size
        self.padding = (kernel_size - 1) // 2
        self.dropout = dropout
        
        self.conv = nn.Conv2d(self.in_dim, self.embed_dim, 1, 1, bias=False)
        self.layers = nn.ModuleList(
            [
                ResidueConvBlock2D(self.embed_dim, 1, 0, dropout) for _ in range(self.num_layers - 1)
            ]
        )
        self.bn = nn.BatchNorm2d(self.embed_dim)
        
    def forward(self, x):  # x: b, n, d
        x = self.conv(x)
        x = self.bn(x)
        
        for layer in self.layers:
            x = layer(x)
            
        return x
    
    
class TimestepEmbedder(nn.Module):
    """
    Embeds scalar timesteps into vector representations.
    """
    def __init__(self, hidden_size, frequency_embedding_size=256):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(frequency_embedding_size, hidden_size, bias=True),
            nn.SiLU(),
            nn.Linear(hidden_size, hidden_size, bias=True),
        )
        self.frequency_embedding_size = frequency_embedding_size

    @staticmethod
    def timestep_embedding(t, dim, max_period=10000):
        """
        Create sinusoidal timestep embeddings.
        :param t: a 1-D Tensor of N indices, one per batch element.
                          These may be fractional.
        :param dim: the dimension of the output.
        :param max_period: controls the minimum frequency of the embeddings.
        :return: an (N, D) Tensor of positional embeddings.
        """
        # https://github.com/openai/glide-text2im/blob/main/glide_text2im/nn.py
        half = dim // 2
        freqs = torch.exp(
            -math.log(max_period) * torch.arange(start=0, end=half, dtype=torch.float32) / half
        ).to(device=t.device)
        args = t[:, None].float() * freqs[None]
        embedding = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)
        if dim % 2:
            embedding = torch.cat([embedding, torch.zeros_like(embedding[:, :1])], dim=-1)
        return embedding

    def forward(self, t):
        t_freq = self.timestep_embedding(t, self.frequency_embedding_size)
        t_emb = self.mlp(t_freq)
        return t_emb
