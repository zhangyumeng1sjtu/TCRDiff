import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from tqdm import tqdm
import numpy as np

from .modules import TransformerLayer, TransformerLayerWithAdaLN, ConvEncoder, LMHead, PairFormerStack, MHCConvLayers, LearnedPositionalEmbedding, LearnedChainPositionalEmbedding, Attention, PairwiseEmbedding, ResidueConvBlock, PairwiseCNN, TimestepEmbedder
from ..utils.tokenizer import PairCDR3Tokenizer, PeptideTokenizer
from .model_utils import sample_from_categorical, stochastic_sample_from_categorical, topk_masking, top_k_top_p_filtering


class PeptideLM(nn.Module):
    def __init__(
        self,
        num_layers = 6,
        dim_head = 64,
        num_attn_heads = 4,
        dim = 256,
        dropout = 0.1,
        max_seq_len = 26, # for single chain
        tokenizer = PeptideTokenizer(),
    ):
        super().__init__()
        
        # Hyper-parameters of Transformer
        self.num_layers = num_layers
        self.dim_head = dim_head
        self.num_attn_heads = num_attn_heads
        self.dim = dim
        self.dropout_prob = dropout
        self.max_seq_len = max_seq_len
        
        # init tokenizer
        self._init_tokenizer(tokenizer)

        # init modules
        self._init_modules()
    
    def _init_tokenizer(self, tokenizer):
        self.tokenizer = tokenizer

        self.alphabet_size = len(self.tokenizer)
        self.padding_idx = self.tokenizer.padding_idx
        self.sep_idx = self.tokenizer.sep_idx
        self.mask_idx = self.tokenizer.mask_idx
        self.bos_idx = self.tokenizer.cls_idx
        self.eos_idx = self.tokenizer.eos_idx
        
    def _init_modules(self):
        
        self.embed_tokens = nn.Embedding(
            self.alphabet_size,
            self.dim,
            padding_idx=self.padding_idx,
        )
        # remove if use rotary embedding
        self.embed_positions = LearnedPositionalEmbedding(
            self.max_seq_len, self.dim, self.padding_idx
        )
        self.transformer = nn.ModuleList(
            [
                TransformerLayer(
                    self.dim,
                    self.dim_head,
                    self.num_attn_heads,
                    2 * self.dim,
                    dropout = self.dropout_prob,
                )
                for _ in range(self.num_layers)
            ]
        )
                
        self.lm_head = LMHead(
            embed_dim = self.dim,
            output_dim = self.alphabet_size,
            weight = self.embed_tokens.weight
        )
    
    def forward(self, tokens, return_repr = False):
        attention_mask = tokens.ne(self.padding_idx).int()
        repr = self.embed_tokens(tokens)
        
        # add positional encoding, remove if use rotary embedding
        repr = self.embed_positions(tokens) + repr
        
        # transformer layers
        for layer in self.transformer:
            repr = layer(repr, mask = attention_mask)
        
        # language model head
        logits = self.lm_head(repr)
        
        if return_repr:
            return logits, repr
        return logits
    
        
class TCRLM(nn.Module):
    def __init__(
        self,
        num_layers = 6,
        dim_head = 64,
        num_attn_heads = 4,
        dim = 256,
        dropout = 0.1,
        max_seq_len = 26, # for single chain
        tokenizer = PairCDR3Tokenizer(),
        use_cdr12_features = True,
        use_timestep_embedding = False,
    ):
        super().__init__()
        
        # Hyper-parameters of Transformer
        self.num_layers = num_layers
        self.dim_head = dim_head
        self.num_attn_heads = num_attn_heads
        self.dim = dim
        self.dropout_prob = dropout
        self.max_seq_len = max_seq_len
        self.use_cdr12_features = use_cdr12_features
        self.use_timestep_embedding = use_timestep_embedding
        
        self.transformer_layer_fn = TransformerLayerWithAdaLN if use_cdr12_features else TransformerLayer
        
        # init tokenizer
        self._init_tokenizer(tokenizer)

        # init modules
        self._init_modules()
    
    def _init_tokenizer(self, tokenizer):
        self.tokenizer = tokenizer

        self.alphabet_size = len(self.tokenizer)
        self.padding_idx = self.tokenizer.padding_idx
        self.sep_idx = self.tokenizer.sep_idx
        self.mask_idx = self.tokenizer.mask_idx
        self.bos_idx = self.tokenizer.cls_idx
        self.eos_idx = self.tokenizer.eos_idx
        
    def _init_modules(self):
        
        self.embed_tokens = nn.Embedding(
            self.alphabet_size,
            self.dim,
            padding_idx=self.padding_idx,
        )
        
        # remove if use rotary embedding
        self.embed_positions_alpha = LearnedChainPositionalEmbedding(
            self.max_seq_len, self.dim, selected_chain_idx=1
        )
        self.embed_positions_beta = LearnedChainPositionalEmbedding(
            self.max_seq_len, self.dim, selected_chain_idx=2
        )
        
        if self.use_timestep_embedding:
            self.timestep_embedding = TimestepEmbedder(hidden_size=self.dim)
        
        self.transformer = nn.ModuleList(
            [
                self.transformer_layer_fn(
                    self.dim,
                    self.dim_head,
                    self.num_attn_heads,
                    2 * self.dim,
                    dropout = self.dropout_prob
                )
                for _ in range(self.num_layers)
            ]
        )
        
        if self.use_cdr12_features:
            self.cdr12_alpha_encoder = ConvEncoder(5, self.dim // 2, max_seq_len=16)
            self.cdr12_beta_encoder = ConvEncoder(5, self.dim // 2, max_seq_len=13)
                
        self.lm_head = LMHead(
            embed_dim = self.dim,
            output_dim = self.alphabet_size,
            weight = self.embed_tokens.weight
        )
    
    def forward(self, cdr3_tokens, chain_token_mask, cdr12_alpha_feat=None, cdr12_beta_feat=None, return_repr = False, time=None):
        attention_mask = cdr3_tokens.ne(self.padding_idx).int()
        repr = self.embed_tokens(cdr3_tokens)

        # remove if use rotary embedding
        repr = self.embed_positions_alpha(chain_token_mask) + self.embed_positions_beta(chain_token_mask) + repr # b, n, d
        
        # add timestep embedding
        if self.use_timestep_embedding:
            repr = self.timestep_embedding(time).unsqueeze(1) + repr
        
        # create conditional embedding of cdr12 sequences
        if self.use_cdr12_features:
            cdr12_alpha_feat = self.cdr12_alpha_encoder(cdr12_alpha_feat) # b, l1, 5 => b, d/2
            cdr12_beta_feat = self.cdr12_beta_encoder(cdr12_beta_feat) # b, l2, 5 => b, d/2
            cdr12_feat = torch.cat((cdr12_alpha_feat, cdr12_beta_feat), dim = 1) # b, d
        
        # transformer layers
        for layer in self.transformer:
            if self.use_cdr12_features:
                repr = layer(repr, cdr12_feat, mask = attention_mask)
            else:
                repr = layer(repr, mask = attention_mask)
        
        # language model head
        logits = self.lm_head(repr)
        
        if return_repr:
            return logits, repr
        return logits


class TCRDPLM(nn.Module):
    
    def __init__(
        self,
        model_config,
        num_diffusion_steps = 100,
        rdm_couple = True,
    ):
        super().__init__()
        
        self.net = TCRLM(**model_config)
        self.num_diffusion_timesteps = num_diffusion_steps
        self.rdm_couple = rdm_couple
        
        self.padding_idx = self.net.padding_idx
        self.sep_idx = self.net.sep_idx
        self.mask_idx = self.net.mask_idx
        self.bos_idx = self.net.bos_idx
        self.eos_idx = self.net.eos_idx
    
    def from_pretrained(self):
        pass

    def q_sample_coupled(self, x_0, t1, t2, maskable_mask):
        # partial mask: True for the part should not be mask
        t1_eq_t2_mask = (t1 == t2)
        t1, t2 = torch.maximum(t1, t2).float(), torch.minimum(t1, t2).float()

        # sample t1
        u = torch.rand_like(x_0, dtype=torch.float)
        t1_mask = (u < (t1 / self.num_diffusion_timesteps)[:, None]) & maskable_mask
        x_t1 = x_0.masked_fill(t1_mask, self.mask_idx)

        # sample t2
        u = torch.rand_like(x_0, dtype=torch.float)
        t2_mask = t1_mask & (u > ((t1 - t2) / t1)[:, None])
        u = torch.rand_like(x_0[t1_eq_t2_mask], dtype=torch.float)
        t2_mask[t1_eq_t2_mask] = (u < (t1[t1_eq_t2_mask] / self.num_diffusion_timesteps)[:, None]) & (maskable_mask[t1_eq_t2_mask])
        x_t2 = x_0.masked_fill(t2_mask, self.mask_idx) 

        return {
            "x_t": torch.cat([x_t1, x_t2], dim=0),
            "t": torch.cat([t1, t2]),
            "mask_mask": torch.cat([t1_mask, t2_mask], dim=0)
        }
    
    def q_sample(self, x_0, t1, maskable_mask):
        # sample t1
        u = torch.rand_like(x_0, dtype=torch.float)
        t1_mask = (u < (t1 / self.num_diffusion_timesteps)[:, None]) & maskable_mask
        x_t1 = x_0.masked_fill(t1_mask, self.mask_idx)

        return {
            "x_t": x_t1,
            "t": t1,
            "mask_mask": t1_mask,
        }
    
    # MLM training
    def forward(self, cdr3_tokens, chain_token_mask, cdr12_alpha_feat, cdr12_beta_feat, return_repr = False, time=None):
        return self.net(cdr3_tokens, chain_token_mask, cdr12_alpha_feat, cdr12_beta_feat, return_repr, time)
    
    def compute_loss(self, batch, partial_masks=None, weighting='constant'):
        # target: unmasked tokens
        target = batch['cdr3_token']
        chain_token_mask = batch['chain_token_mask']
        cdr12_alpha_feat = batch['cdr12_alpha_feat']
        cdr12_beta_feat = batch['cdr12_beta_feat']
        
        # couple
        t1, t2 = torch.randint(
            1, self.num_diffusion_timesteps + 1,
            (2 * target.size(0), ),
            device=target.device
        ).chunk(2)

        if self.rdm_couple:
            x_t, t, loss_mask = list(
                self.q_sample_coupled(
                    target, t1, t2,
                    maskable_mask=self.get_non_special_sym_mask(target, partial_masks)
                ).values()
            )
            target = target.repeat(2,1)
            chain_token_mask = chain_token_mask.repeat(2,1)
            cdr12_alpha_feat = cdr12_alpha_feat.repeat(2,1,1)
            cdr12_beta_feat = cdr12_beta_feat.repeat(2,1,1)
        else:
            x_t, t, loss_mask = list(
                self.q_sample(
                    target, t1,
                    maskable_mask=self.get_non_special_sym_mask(target, partial_masks)
                ).values()
            )

        # add timestep embedding 
        if self.net.use_timestep_embedding:
            logits = self.forward(x_t, chain_token_mask, cdr12_alpha_feat, cdr12_beta_feat, time=t)
        else:
            logits = self.forward(x_t, chain_token_mask, cdr12_alpha_feat, cdr12_beta_feat)
        
        num_timesteps = self.num_diffusion_timesteps
        weight = {
            "linear": (num_timesteps - (t - 1)),  # num_timesteps * (1 - (t-1)/num_timesteps)
            "constant": num_timesteps * torch.ones_like(t)
        }[weighting][:, None].float() / num_timesteps
        
        return logits, target, loss_mask, weight
    
    def forward_encoder(self, batch, **kwargs):
        return {}
    
    def get_non_special_sym_mask(self, output_tokens, partial_masks=None):
        non_special_sym_mask = (
            output_tokens.ne(self.padding_idx) &
            output_tokens.ne(self.bos_idx) &
            output_tokens.ne(self.eos_idx) &
            output_tokens.ne(self.sep_idx)
        )
        if partial_masks is not None:
            non_special_sym_mask &= (~partial_masks)
        return non_special_sym_mask

    def initialize_output_tokens(self, batch, partial_masks=None, use_draft_seq=False):
        if use_draft_seq:
            initial_output_tokens = batch['cdr3_token']
            initial_output_scores = torch.zeros(
                *initial_output_tokens.size(), device=initial_output_tokens.device
            )
        else:
            tokens = batch['cdr3_token']
            output_mask = self.get_non_special_sym_mask(tokens, partial_masks=partial_masks)

            output_tokens = tokens.masked_fill(output_mask, self.mask_id)
            output_scores = torch.zeros_like(output_tokens, dtype=torch.float)

            return output_tokens, output_scores
        
        return initial_output_tokens, initial_output_scores
    
    def forward_decoder(self, prev_decoder_out, chain_token_mask, cdr12_alpha_feat, cdr12_beta_feat, partial_masks=None, sampling_strategy='gumbel_argmax'):
        output_tokens = prev_decoder_out['output_tokens'].clone()
        output_scores = prev_decoder_out['output_scores'].clone()
        step, max_step = prev_decoder_out['step'], prev_decoder_out['max_step']
        temperature = prev_decoder_out['temperature']
        history = prev_decoder_out['history']

        output_masks = self.get_non_special_sym_mask(output_tokens, partial_masks=partial_masks)
        
        if self.net.use_timestep_embedding:
            time = (1 - step / max_step) * self.num_diffusion_timesteps
            logits, last_hidden_state = self.forward(output_tokens, chain_token_mask, cdr12_alpha_feat, cdr12_beta_feat, return_repr=True, time=time)
        else:
            logits, last_hidden_state = self.forward(output_tokens, chain_token_mask, cdr12_alpha_feat, cdr12_beta_feat, return_repr=True)
            
        logits[..., self.mask_idx] = -math.inf
        logits[..., self.padding_idx] = -math.inf
        logits[..., self.bos_idx] = -math.inf
        logits[..., self.eos_idx] = -math.inf
        logits[..., self.sep_idx] = -math.inf
        
        if sampling_strategy == 'vanilla':
            _tokens, _scores = sample_from_categorical(logits, temperature=temperature)
        elif sampling_strategy == 'argmax':
            _scores, _tokens = logits.max(-1)
        elif sampling_strategy == 'gumbel_argmax':
            noise_scale = 1.0
            _tokens, _scores = stochastic_sample_from_categorical(logits, temperature=temperature, noise_scale=noise_scale)
        else:
            raise NotImplementedError
        
        output_tokens.masked_scatter_(output_masks, _tokens[output_masks])
        output_scores.masked_scatter_(output_masks, _scores[output_masks])
        
        history.append(output_tokens.clone())

        return dict(
            output_tokens=output_tokens,
            output_scores=output_scores,
            step=step + 1,
            max_step=max_step,
            history=history,
            hidden_states=last_hidden_state
        )
    
    def generate(self, batch, tokenizer=None, max_iter=None, temperature=None, partial_masks=None, sampling_strategy='gumbel_argmax', use_draft_seq=False):
        
        tokenizer = tokenizer 
        max_iter = max_iter
        temperature = temperature

        initial_output_tokens, initial_output_scores = self.initialize_output_tokens(batch, partial_masks=partial_masks, use_draft_seq=use_draft_seq)
        prev_decoder_out = dict(
            output_tokens=initial_output_tokens,
            output_scores=initial_output_scores,
            output_masks=None,
            step=0,
            max_step=max_iter,
            history=[initial_output_tokens.clone()],
            temperature=temperature,
        )

        prev_decoder_out['output_masks'] = self.get_non_special_sym_mask(
            prev_decoder_out['output_tokens'], partial_masks=partial_masks
        )

        for step in tqdm(range(max_iter), desc='Decoding'):
            with torch.no_grad():
                decoder_out = self.forward_decoder(
                    prev_decoder_out=prev_decoder_out,
                    chain_token_mask=batch['chain_token_mask'],
                    cdr12_alpha_feat=batch['cdr12_alpha_feat'],
                    cdr12_beta_feat=batch['cdr12_beta_feat'],
                    partial_masks=partial_masks,
                    sampling_strategy=sampling_strategy
                )

            output_tokens = decoder_out['output_tokens']
            output_scores = decoder_out['output_scores']
            
            # re-mask skeptical parts of low confidence
            non_special_sym_mask = self.get_non_special_sym_mask(
                prev_decoder_out['output_tokens'], partial_masks=partial_masks
            )
            
            output_masks, result_tokens, result_scores = self._reparam_decoding(
                output_tokens=prev_decoder_out['output_tokens'].clone(),
                output_scores=prev_decoder_out['output_scores'].clone(),
                cur_tokens=output_tokens.clone(),
                cur_scores=output_scores.clone(),
                decoding_strategy='reparam-uncond-deterministic-linear', # 'reparam-uncond-stochastic1.0-linear'
                xt_neq_x0=prev_decoder_out['output_masks'],
                non_special_sym_mask=non_special_sym_mask,
                t=step + 1,
                max_step=max_iter,
                noise=self.mask_idx,
            )
            
            prev_decoder_out.update(output_masks=output_masks)
            output_tokens = result_tokens
            output_scores = result_scores

            prev_decoder_out.update(
                output_tokens=output_tokens,
                output_scores=output_scores,
                step=step + 1,
                history=decoder_out['history']
            )

        decoder_out = prev_decoder_out
        return decoder_out['output_tokens'], decoder_out['output_scores']
    
    def _reparam_decoding(
        self,
        output_tokens,
        output_scores,
        cur_tokens,
        cur_scores,
        decoding_strategy,
        xt_neq_x0,
        non_special_sym_mask,
        t,
        max_step,
        noise,
    ):
        """
            This function is used to perform reparameterized decoding.
        """
        # output_tokens: [B, N]
        # output_scores: [B, N]
        # cur_tokens: [B, N]
        # cur_scores: [B, N]
        # xt_neq_x0: equivalent to not_b_t [B, N]
        # non_special_sym_mask: [B, N]
        # noise: either [B, N] or scalar (if using the mask noise)

        # decoding_strategy needs to take the form of "reparam-<conditioning>-<topk_mode>-<schedule>"
        _, condition, topk_mode, schedule = decoding_strategy.split("-")

        # first set the denoising rate according to the schedule
        if schedule == "linear":
            rate = 1 - t / max_step
        elif schedule == "cosine":
            rate = np.cos(t / max_step * np.pi * 0.5)
        else:
            raise NotImplementedError

        # compute the cutoff length for denoising top-k positions
        cutoff_len = (
            non_special_sym_mask.sum(1, keepdim=True).type_as(output_scores) * rate
        ).long()
        # set the scores of special symbols to a large value so that they will never be selected
        _scores_for_topk = cur_scores.masked_fill(~non_special_sym_mask, 1000.0)
        
        to_be_resample = []
        for i, seq in enumerate(cur_tokens):
            most_token_dict = {}
            most_token_num = -1
            for j, token in enumerate(seq):
                token = int(token)
                if token == self.padding_idx:
                    continue
                if token not in most_token_dict:
                    most_token_dict[token] = [j]
                else:
                    most_token_dict[token].append(j)
                if len(most_token_dict[token]) > most_token_num:
                    most_token_num = len(most_token_dict[token])
            if most_token_num > len(seq) * 0.25:
                to_be_resample.append(i)
                
        # the top-k selection can be done in two ways: stochastic by injecting Gumbel noise or deterministic
        if topk_mode.startswith("stochastic"):
            noise_scale = float(topk_mode.replace("stochastic", ""))
            lowest_k_mask = topk_masking(_scores_for_topk, cutoff_len, stochastic=True, temp=noise_scale * rate)
        elif topk_mode == "deterministic":
            lowest_k_mask = topk_masking(_scores_for_topk, cutoff_len, stochastic=False)
            if len(to_be_resample) > 0:
                noise_scale = 1.5
                lowest_k_mask[to_be_resample] = topk_masking(_scores_for_topk[to_be_resample], cutoff_len[to_be_resample], 
                                                             stochastic=True, temp=noise_scale * rate)
        else:
            raise NotImplementedError

        # Various choices to generate v_t := [v1_t, v2_t].
        # Note that
        #   v1_t governs the outcomes of tokens where b_t = 1,
        #   v2_t governs the outcomes of tokens where b_t = 0.

        # #### the `uncond` mode ####
        # In our reparameterized decoding,
        # both v1_t and v2_t can be fully determined by the current token scores .

        # #### the `cond` mode ####
        # However, we can also impose some conditional constraints on v1_t so that
        # the decoding can be performed in a more conservative manner.
        # For example, we can set v1_t = 0 only when
        # (the newly output tokens are the same as previous denoised results, AND
        # the current token score becomes lower, AND
        # the current token score is not in the top-k share among all tokens).
        if condition == "cond":
            not_v1_t = (cur_tokens == output_tokens) & (cur_scores < output_scores) & lowest_k_mask
        elif condition == "uncond":
            not_v1_t = lowest_k_mask
        else:
            raise NotImplementedError

        # for b_t = 0, the token is set to noise if it is in the lowest k scores.
        not_v2_t = lowest_k_mask

        last_mask_position = xt_neq_x0
        masked_to_noise = (~xt_neq_x0 & not_v1_t) | (xt_neq_x0 & not_v2_t)
        if isinstance(noise, torch.Tensor):
            output_tokens.masked_scatter_(masked_to_noise, noise[masked_to_noise])
        elif isinstance(noise, (int, float)):
            output_tokens.masked_fill_(masked_to_noise, noise)
        else:
            raise NotImplementedError("noise should be either a tensor or a scalar")
        output_scores.masked_fill_(masked_to_noise, -math.inf)

        masked_to_x0 = xt_neq_x0 & ~not_v2_t
        output_tokens.masked_scatter_(masked_to_x0, cur_tokens[masked_to_x0])
        output_scores.masked_scatter_(masked_to_x0, cur_scores[masked_to_x0])
        assert ((masked_to_x0 & last_mask_position) == masked_to_x0).all()
        # b_{t} = (b_{t+1} & u_t) | v_t
        # For convenience, save the NOT of b_t for the next iteration
        # NOT_b_{t} = (NOT_b_{t+1} | not_v1_t) & not_v2_t
        #
        # # When condition is 'uncond', the not_v1_t is equal to not_v2_t, the new_xt_neq_x0 is always equal to not_v1/v2_t
        new_xt_neq_x0 = (xt_neq_x0 | not_v1_t) & not_v2_t
        assert (new_xt_neq_x0 == not_v2_t).all()
        return new_xt_neq_x0, output_tokens, output_scores
       
# Peptide-MHC model: pre-trained peptide LM + CNN (MHC 175aa) => 
# peptide + MHC-pseudo module [binding pairformer + structure pairformer] => 
# binding affinity/eluted ligand + reconstruct masked peptide positions
class PeptideMHCPairFormer(nn.Module):
    def __init__(
        self,
        num_transformer_layers = 6,
        transformer_attn_dim_head = 32,
        transformer_attn_heads = 8,
        dim_single = 256,
        dim_pairwise = 128,
        pair_bias_attn_dim_head = 32,
        pair_bias_attn_heads = 8,
        tri_attn_dim_head = 32,
        tri_attn_heads = 4,
        dropout = 0.1,
        max_pep_len = 26,
        dim_mhc_in = 20,
        max_mhc_len = 175,
        num_cnn_layers = 4,
        pair_method = 'attention',
        tokenizer = PeptideTokenizer(),
    ):
        super().__init__()
        
        self.peptide_model = PeptideLM(
            num_transformer_layers, transformer_attn_dim_head, transformer_attn_heads, 
            dim_single, dropout, max_pep_len, tokenizer
        )
        
        self.mhc_model = MHCConvLayers(
            num_cnn_layers, dim_mhc_in, dim_single, max_mhc_len,
            kernel_size=3, dropout=dropout
        )
        
        # pairwise modeling 
        self.pairwsie_pmhc = PairwiseEmbedding(dim_single, dim_pairwise, method=pair_method) # method = 'attention' or 'concat'
        
        self.layernorm = nn.LayerNorm(dim_single)
        self.pmhc_binding_pairformer = PairFormerStack(
            dim_single = dim_single, dim_pairwise = dim_pairwise, depth = 1, # depth = 2
            pair_bias_attn_dim_head = pair_bias_attn_dim_head,
            pair_bias_attn_heads = pair_bias_attn_heads,
            dropout_row_prob = dropout,
            pairwise_block_kwargs = {
                'tri_attn_dim_head': tri_attn_dim_head,
                'tri_attn_heads': tri_attn_heads,
                'dropout_row_prob': dropout,
                'dropout_col_prob': dropout,
            }
        )
        
        # binding prediction
        self.mhc_attn = nn.Sequential(
            nn.Linear(dim_single, dim_single),
            nn.Tanh(),
            nn.Linear(dim_single, 1)
        )
        self.binding_head = nn.Sequential(
            nn.Linear(dim_single * 2, dim_single),
            nn.ReLU(),
            nn.Dropout(p=dropout),
            nn.Linear(dim_single, 1),
            nn.Sigmoid()
        )
        
        # mlm prediction
        self.lm_head = LMHead(
            embed_dim = dim_single,
            output_dim = len(tokenizer),
            weight = self.peptide_model.embed_tokens.weight
        )
        self.tokenizer = tokenizer
        
    def load_pretrained_peptide_model(self, pretrained_model_path):
        # Load the pre-trained epitope LM model and freeze weights (try lora or fintune?)
        self.peptide_model.load_state_dict(torch.load(pretrained_model_path, map_location='cpu', weights_only=True))
        for param in self.peptide_model.parameters():
            param.requires_grad = False
    
    def forward(self, peptide, mhc, mhc_pseudo_mask, return_repr = False):
        
        # obtain pre-trained peptide embeddings
        _, pep_single_repr = self.peptide_model(peptide, return_repr = True) # b, n => b, n, ds + b, n, n, d
        pep_attention_mask = peptide.ne(self.tokenizer.padding_idx).int()
        
        # get full-length mhc feature embeddings and extract pseudo sequences
        mhc_single_repr = self.mhc_model(mhc) # b, M, 20 => b, M, ds
        mhc_single_repr = mhc_single_repr[mhc_pseudo_mask.bool()].view(mhc_single_repr.size(0), 34, mhc_single_repr.size(2)) # b, M, ds => b, m, ds
        mhc_attention_mask = torch.ones((mhc_single_repr.size(0), mhc_single_repr.size(1)), device = mhc_single_repr.device).int()
        
        pmhc_single_repr = torch.cat((pep_single_repr, mhc_single_repr), dim=1) # b, n + m, ds
        pmhc_attention_mask = torch.cat((pep_attention_mask, mhc_attention_mask), dim = 1) # b, n + m
        pmhc_pairwise_repr = self.pairwsie_pmhc(pmhc_single_repr.clone(), pmhc_single_repr.clone()) # b, n + m, ds => b, n + m, dp
        
        
        # compute binding and structure repr
        binding_single_repr, binding_pairwise_repr = self.pmhc_binding_pairformer(
            single_repr = pmhc_single_repr,
            pairwise_repr = pmhc_pairwise_repr,
            mask = pmhc_attention_mask
        )
        
        if return_repr:
            return binding_single_repr, binding_pairwise_repr, pmhc_attention_mask
        
        pep_binding_repr, mhc_binding_repr = binding_single_repr[:, 0, :], binding_single_repr[:, pep_single_repr.size(1):, :]
        
        mhc_attn = F.softmax(self.mhc_attn(mhc_binding_repr), dim=1) # b, m, d => b, m, 1
        mhc_pooled = torch.bmm(mhc_binding_repr.transpose(1,2), mhc_attn).squeeze(2) # b, d
        
        pmhc_binding_repr = torch.cat([pep_binding_repr.squeeze(1), mhc_pooled], dim=1) # b, 2d
        out = self.binding_head(pmhc_binding_repr)
        
        pep_single_repr += self.layernorm(binding_single_repr[:, :pep_single_repr.size(1), :])
        logits = self.lm_head(pep_single_repr)

        return out, logits
    

# use cross attn to connect peptide and mhc features
class PeptideMHCModel(nn.Module):
    def __init__(
        self,
        num_transformer_layers = 6,
        transformer_attn_dim_head = 32,
        transformer_attn_heads = 8,
        dim_single = 256,
        cross_attn_dim_head = 32,
        cross_attn_heads = 8,
        dropout = 0.1,
        max_pep_len = 26,
        dim_mhc_in = 20,
        max_mhc_len = 175,
        num_cnn_layers = 4,
        tokenizer = PeptideTokenizer(),
    ):
        super().__init__()
        
        self.peptide_model = PeptideLM(
            num_transformer_layers, transformer_attn_dim_head, transformer_attn_heads, 
            dim_single, dropout, max_pep_len, tokenizer
        )
        
        self.mhc_model = MHCConvLayers(
            num_cnn_layers, dim_mhc_in, dim_single, max_mhc_len,
            kernel_size=3, dropout=dropout
        )
        
        self.peptide_cross_attn = Attention(
            dim=dim_single,
            dim_head=cross_attn_dim_head,
            heads=cross_attn_heads,
            dropout=dropout,
            gate_output=False)
        
        self.mhc_cross_attn = Attention(
            dim=dim_single,
            dim_head=cross_attn_dim_head,
            heads=cross_attn_heads,
            dropout=dropout,
            gate_output=False)
        
        self.layernorm = nn.LayerNorm(dim_single)
        # binding prediction
        self.mhc_attn = nn.Sequential(
            nn.Linear(dim_single, dim_single),
            nn.Tanh(),
            nn.Linear(dim_single, 1)
        )
        self.binding_head = nn.Sequential(
            nn.Linear(dim_single * 2, dim_single),
            nn.ReLU(),
            nn.Dropout(p=dropout),
            nn.Linear(dim_single, 1),
            nn.Sigmoid()
        )
        
        # mlm prediction
        self.lm_head = LMHead(
            embed_dim = dim_single,
            output_dim = len(tokenizer),
            weight = self.peptide_model.embed_tokens.weight
        )
        self.tokenizer = tokenizer
    
    def load_pretrained_peptide_model(self, pretrained_model_path):
        # Load the pre-trained epitope LM model and freeze weights (try lora or fintune?)
        self.peptide_model.load_state_dict(torch.load(pretrained_model_path, map_location='cpu', weights_only=True))
        for param in self.peptide_model.parameters():
            param.requires_grad = False
            
    def forward(self, peptide, mhc, mhc_pseudo_mask, return_repr = False):
        
        # obtain pre-trained peptide embeddings
        _, pep_single_repr = self.peptide_model(peptide, return_repr = True) # b, n => b, n, ds + b, n, n, d
        pep_attention_mask = peptide.ne(self.tokenizer.padding_idx).int()
        
        # get full-length mhc feature embeddings and extract pseudo sequences
        mhc_single_repr = self.mhc_model(mhc) # b, M, 20 => b, M, ds
        mhc_single_repr = mhc_single_repr[mhc_pseudo_mask.bool()].view(mhc_single_repr.size(0), 34, mhc_single_repr.size(2)) # b, M, ds => b, m, ds
        mhc_attention_mask = torch.ones((mhc_single_repr.size(0), mhc_single_repr.size(1)), device = mhc_single_repr.device).int()
        
        pep_binding_repr = self.peptide_cross_attn(pep_single_repr, mask=mhc_attention_mask, context=mhc_single_repr)
        mhc_binding_repr = self.mhc_cross_attn(mhc_single_repr, mask=pep_attention_mask, context=pep_single_repr)
        
        if return_repr:
            return torch.cat((pep_binding_repr, mhc_binding_repr), dim=1), torch.cat((pep_attention_mask, mhc_attention_mask), dim=1)
         
        mhc_attn = F.softmax(self.mhc_attn(mhc_binding_repr), dim=1) # b, m, d => b, m, 1
        mhc_pooled = torch.bmm(mhc_binding_repr.transpose(1,2), mhc_attn).squeeze(2) # b, d
        
        pmhc_binding_repr = torch.cat([pep_binding_repr[:, 0, :].squeeze(1), mhc_pooled], dim=1) # b, 2d
        out = self.binding_head(pmhc_binding_repr)
        
        logits =  self.lm_head(pep_single_repr + self.layernorm(pep_binding_repr))
        # logits = self.lm_head(pep_binding_repr)
        
        return out, logits
    
    
class TCRpMHCPairFormer(nn.Module):
    def __init__(
        self,
        tcr_model_config,
        pmhc_model_config,
        dim_single = 256,
        dim_pairwise = 64,
        pair_bias_attn_dim_head = 32,
        pair_bias_attn_heads = 8,
        tri_attn_dim_head = 16,
        tri_attn_heads = 4,
        self_attn_dim_head = 32,
        self_attn_heads = 8,
        dropout = 0.1,
        pair_method = 'attention',
        use_pmhc_struc_feat = False,
        num_pmhc_struc_layers = 2,
    ):
        super().__init__()

        self.tcr_model = TCRDPLM(tcr_model_config)
        self.pmhc_model = PeptideMHCPairFormer(**pmhc_model_config)
        
        self.pairwise_tcr = PairwiseEmbedding(dim_single, dim_pairwise, pair_method)
        self.pairwise_tcr2pmhc = PairwiseEmbedding(dim_single, dim_pairwise, pair_method)
        self.pairwise_pmhc2tcr = PairwiseEmbedding(dim_single, dim_pairwise, pair_method)
        self.dim_pairwise = dim_pairwise
        
        self.use_pmhc_struc_feat = use_pmhc_struc_feat
        if self.use_pmhc_struc_feat:
            self.pmhc_struc_cnn = PairwiseCNN(num_layers=num_pmhc_struc_layers, in_dim=3, embed_dim=dim_pairwise, kernel_size=1, dropout=dropout)
        
        self.binding_pairformer = PairFormerStack(
            dim_single = dim_single, dim_pairwise = dim_pairwise, depth = 1, # depth = 2
            pair_bias_attn_dim_head = pair_bias_attn_dim_head,
            pair_bias_attn_heads = pair_bias_attn_heads,
            dropout_row_prob = dropout,
            pairwise_block_kwargs = {
                'tri_attn_dim_head': tri_attn_dim_head,
                'tri_attn_heads': tri_attn_heads,
                'dropout_row_prob': dropout,
                'dropout_col_prob': dropout,
            }
        )
        
        self.tcr_self_attn = Attention( 
            dim=dim_single,
            dim_head=self_attn_dim_head,
            heads=self_attn_heads,
            dropout=dropout,
            gate_output=False,
            use_rotary_embedding=False # if not use rotary embedding, set use_rotary_embedding=False
        )
        self.tcr_layer_norm = nn.LayerNorm(dim_single)
        
        self.pmhc_self_attn = Attention( 
            dim=dim_single,
            dim_head=self_attn_dim_head,
            heads=self_attn_heads,
            dropout=dropout,
            gate_output=False,
            use_rotary_embedding=False
        )
        self.pmhc_layer_norm = nn.LayerNorm(dim_single)
        
        self.binding_head = nn.Sequential(
            nn.Linear(dim_single * 2, dim_single),
            nn.ReLU(),
            nn.Dropout(p=dropout),
            nn.Linear(dim_single, 1),
            nn.Sigmoid()
        )
        
    def load_pretrained_weights(self, pretrained_tcr_model_path, pretrained_pmhc_model_path):
        # load pre-trained tcr and pmhc models and freeze weights
        if pretrained_tcr_model_path is not None:
            self.tcr_model.load_state_dict(torch.load(pretrained_tcr_model_path, map_location='cpu', weights_only=True))
            for param in self.tcr_model.parameters():
                param.requires_grad = False
                
        if pretrained_pmhc_model_path is not None:
            self.pmhc_model.load_state_dict(torch.load(pretrained_pmhc_model_path, map_location='cpu', weights_only=True))
            for param in self.pmhc_model.parameters():
                param.requires_grad = False
            
    def forward(self, batch):
        # input features: cdr3_tokens, chain_token_mask, cdr12_alpha_feat, cdr12_beta_feat, peptide, mhc, mhc_pseudo_mask
        _, tcr_single_repr = self.tcr_model(batch['cdr3_token'], batch['chain_token_mask'], batch['cdr12_alpha_feat'], batch['cdr12_beta_feat'], return_repr=True)
        pmhc_single_repr, pmhc_pairwise_repr, pmhc_attention_mask = self.pmhc_model(batch['peptide_token'], batch['mhc_embedding'], batch['mhc_pseudo_mask'], return_repr=True)
        tcr_attention_mask = batch['cdr3_token'].ne(self.tcr_model.padding_idx).int()
        
        if self.use_pmhc_struc_feat:
            pmhc_struc_repr = self.pmhc_struc_cnn(batch['pmhc_struc_feat'])
            pmhc_pairwise_repr = pmhc_pairwise_repr.clone() + pmhc_struc_repr.permute(0, 2, 3, 1)
        
        tcr_pairwise_repr = self.pairwise_tcr(tcr_single_repr.clone(), tcr_single_repr.clone())
        
        tcr2pmhc_pairwise_repr = self.pairwise_tcr2pmhc(tcr_single_repr.clone(), pmhc_single_repr.clone())
        pmhc2tcr_pairwise_repr = self.pairwise_pmhc2tcr(pmhc_single_repr.clone(), tcr_single_repr.clone())
        
        # concatenate tcr-pmhc representations and attention masks
        single_repr = torch.cat([tcr_single_repr, pmhc_single_repr], dim=1)
        attention_mask = torch.cat([tcr_attention_mask, pmhc_attention_mask], dim=1)
        
        # initialize pairwise repr of tcr pmhc
        pairwise_repr = torch.cat([
            torch.cat([tcr_pairwise_repr, tcr2pmhc_pairwise_repr], dim=2),
            torch.cat([pmhc2tcr_pairwise_repr, pmhc_pairwise_repr], dim=2)
        ], dim=1)
        
        # tcr pmhc pairformer modeling
        binding_single_repr, _ = self.binding_pairformer(
            single_repr = single_repr,
            pairwise_repr = pairwise_repr,
            mask = attention_mask
        )
        
        tcr_single_repr += self.tcr_layer_norm(self.tcr_self_attn(
            binding_single_repr[:, :tcr_single_repr.size(1), :], mask = tcr_attention_mask
        ))
        pmhc_single_repr += self.pmhc_layer_norm(self.pmhc_self_attn(
            binding_single_repr[:, tcr_single_repr.size(1):, :], mask = pmhc_attention_mask
        ))
        
        # remove self-attention
        # tcr_single_repr += self.tcr_layer_norm(binding_single_repr[:, :tcr_single_repr.size(1), :])
        # pmhc_single_repr += self.pmhc_layer_norm(binding_single_repr[:, tcr_single_repr.size(1):, :])
        
        # concatenate <cls> representations of tcr and pmhc
        out = self.binding_head(torch.cat([tcr_single_repr[:, 0, :], pmhc_single_repr[:, 0, :]], dim=1))    
        return out
    

# EPACT like model architecture (project tcr and pmhc repr to a co-embedding space)
class TCRpMHCCoembeddingModel(nn.Module):
    def __init__(
        self,
        tcr_model_config,
        pmhc_model_config,
        dim = 256,
        self_attn_dim_head = 32,
        self_attn_heads = 8,
        num_conv_layers = 2,
        dropout = 0.1,
    ):
        super().__init__()

        self.tcr_model = TCRDPLM(tcr_model_config)
        # self.pmhc_model = PeptideMHCPairFormer(**pmhc_model_config)
        self.pmhc_model = PeptideMHCModel(**pmhc_model_config)
        
        self.tcr_self_attn = Attention(
            dim=dim,
            dim_head=self_attn_dim_head,
            heads=self_attn_heads,
            dropout=dropout,
            gate_output=False,
            use_rotary_embedding=False
        )
        self.tcr_layer_norm = nn.LayerNorm(dim)    
        self.pmhc_self_attn = Attention(
            dim=dim,
            dim_head=self_attn_dim_head,
            heads=self_attn_heads,
            dropout=dropout,
            gate_output=False,
            use_rotary_embedding=False
        )
        self.pmhc_layer_norm = nn.LayerNorm(dim)
        
        self.tcr_conv_layers = nn.ModuleList(
            [
                ResidueConvBlock(embed_dim=dim, kernel_size=1, padding=0, dropout=dropout) 
                for _ in range(num_conv_layers)
            ]
        )
        self.pmhc_conv_layers = nn.ModuleList(
            [
                ResidueConvBlock(embed_dim=dim, kernel_size=1, padding=0, dropout=dropout) 
                for _ in range(num_conv_layers)
            ]
        )
        
        self.tcr_projector = nn.Sequential(
            nn.Linear(dim, dim), nn.BatchNorm1d(dim), nn.ReLU(), nn.Linear(dim, dim)
        )
        self.pmhc_projector = nn.Sequential(
            nn.Linear(dim, dim), nn.BatchNorm1d(dim), nn.ReLU(), nn.Linear(dim, dim)
        )

        self.binding_head = nn.Sequential(
            nn.Linear(dim * 2, dim),
            nn.ReLU(),
            nn.Dropout(p=dropout),
            nn.Linear(dim, 1),
            nn.Sigmoid()
        )
    
    def load_pretrained_weights(self, pretrained_tcr_model_path, pretrained_pmhc_model_path):
        # load pre-trained tcr and pmhc models and freeze weights
        self.tcr_model.load_state_dict(torch.load(pretrained_tcr_model_path, map_location='cpu', weights_only=True))
        self.pmhc_model.load_state_dict(torch.load(pretrained_pmhc_model_path, map_location='cpu', weights_only=True))

        for param in self.tcr_model.parameters():
            param.requires_grad = False
        for param in self.pmhc_model.parameters():
            param.requires_grad = False
            
    def forward(self, batch):
        # input features: cdr3_tokens, chain_token_mask, cdr12_alpha_feat, cdr12_beta_feat, peptide, mhc, mhc_pseudo_mask
        _, tcr_single_repr = self.tcr_model(batch['cdr3_token'], batch['chain_token_mask'], batch['cdr12_alpha_feat'], batch['cdr12_beta_feat'], return_repr=True)
        # pmhc_single_repr, _, pmhc_attention_mask = self.pmhc_model(batch['peptide_token'], batch['mhc_embedding'], batch['mhc_pseudo_mask'], return_repr=True)
        pmhc_single_repr, pmhc_attention_mask = self.pmhc_model(batch['peptide_token'], batch['mhc_embedding'], batch['mhc_pseudo_mask'], return_repr=True) # use cross-attn pmhc model
        tcr_attention_mask = batch['cdr3_token'].ne(self.tcr_model.padding_idx).int()
        
        # layernorm and residual connection
        tcr_single_repr += self.tcr_layer_norm(self.tcr_self_attn(
            tcr_single_repr.clone(), mask = tcr_attention_mask
        ))
        pmhc_single_repr += self.pmhc_layer_norm(self.pmhc_self_attn(
            pmhc_single_repr.clone(), mask = pmhc_attention_mask
        ))
        
        tcr_single_repr = tcr_single_repr.transpose(1,2) # B, E, L1
        pmhc_single_repr = pmhc_single_repr.transpose(1,2) # B, E, L2
        
        for layer in self.tcr_conv_layers:
            tcr_single_repr = layer(tcr_single_repr)
        for layer in self.pmhc_conv_layers:
            pmhc_single_repr = layer(pmhc_single_repr)
        
        tcr_projection = tcr_single_repr[:, :, 0]
        pmhc_projection = pmhc_single_repr[:, :, 0]
        
        embed = torch.cat([tcr_projection, pmhc_projection], dim=1) 
        logits = self.binding_head(embed)
        
        tcr_projection = self.tcr_projector(tcr_projection)
        pmhc_projection = self.pmhc_projector(pmhc_projection)
        dist = F.cosine_similarity(tcr_projection, pmhc_projection, dim=-1)
    
        output = {'logits': logits, 'dist': dist, 'projection': (tcr_projection, pmhc_projection)}
            
        return output
    
    
class TCRpMHCCoembeddingPairFormer(nn.Module):
    def __init__(
        self,
        tcr_model_config,
        pmhc_model_config,
        dim_single = 256,
        dim_pairwise = 64,
        pair_bias_attn_dim_head = 32,
        pair_bias_attn_heads = 8,
        tri_attn_dim_head = 16,
        tri_attn_heads = 4,
        self_attn_dim_head = 32,
        self_attn_heads = 8,
        dropout = 0.1,
        pair_method = 'attention',
        use_pmhc_struc_feat = False,
        num_pmhc_struc_layers = 2, # 3
    ):
        super().__init__()

        self.tcr_model = TCRDPLM(tcr_model_config)
        self.pmhc_model = PeptideMHCPairFormer(**pmhc_model_config)
        
        self.pairwise_tcr = PairwiseEmbedding(dim_single, dim_pairwise, pair_method)
        self.pairwise_tcr2pmhc = PairwiseEmbedding(dim_single, dim_pairwise, pair_method)
        self.pairwise_pmhc2tcr = PairwiseEmbedding(dim_single, dim_pairwise, pair_method)
        self.dim_pairwise = dim_pairwise
        
        self.use_pmhc_struc_feat = use_pmhc_struc_feat
        if self.use_pmhc_struc_feat:
            self.pmhc_struc_cnn = PairwiseCNN(num_layers=num_pmhc_struc_layers, in_dim=3, embed_dim=dim_pairwise, kernel_size=1, dropout=dropout)
        
        self.binding_pairformer = PairFormerStack(
            dim_single = dim_single, dim_pairwise = dim_pairwise, depth = 1,
            pair_bias_attn_dim_head = pair_bias_attn_dim_head,
            pair_bias_attn_heads = pair_bias_attn_heads,
            dropout_row_prob = dropout,
            pairwise_block_kwargs = {
                'tri_attn_dim_head': tri_attn_dim_head,
                'tri_attn_heads': tri_attn_heads,
                'dropout_row_prob': dropout,
                'dropout_col_prob': dropout,
            }
        )
        
        self.tcr_self_attn = Attention(
            dim=dim_single,
            dim_head=self_attn_dim_head,
            heads=self_attn_heads,
            dropout=dropout,
            gate_output=False,
            use_rotary_embedding=False
        )
        self.tcr_layer_norm = nn.LayerNorm(dim_single)
        
        self.pmhc_self_attn = Attention(
            dim=dim_single,
            dim_head=self_attn_dim_head,
            heads=self_attn_heads,
            dropout=dropout,
            gate_output=False,
            use_rotary_embedding=False
        )
        self.pmhc_layer_norm = nn.LayerNorm(dim_single)
        
        self.tcr_projector = nn.Sequential(
            nn.Linear(dim_single, dim_single), nn.BatchNorm1d(dim_single), nn.ReLU(), nn.Linear(dim_single, dim_single)
        )
        self.pmhc_projector = nn.Sequential(
            nn.Linear(dim_single, dim_single), nn.BatchNorm1d(dim_single), nn.ReLU(), nn.Linear(dim_single, dim_single)
        )
        
        self.binding_head = nn.Sequential(
            nn.Linear(dim_single * 2, dim_single),
            nn.ReLU(),
            nn.Dropout(p=dropout),
            nn.Linear(dim_single, 1),
            nn.Sigmoid()
        )
    
    def load_pretrained_weights(self, pretrained_tcr_model_path, pretrained_pmhc_model_path):
        # load pre-trained tcr and pmhc models and freeze weights
        self.tcr_model.load_state_dict(torch.load(pretrained_tcr_model_path, map_location='cpu', weights_only=True))
        self.pmhc_model.load_state_dict(torch.load(pretrained_pmhc_model_path, map_location='cpu', weights_only=True))

        for param in self.tcr_model.parameters():
            param.requires_grad = False
        for param in self.pmhc_model.parameters():
            param.requires_grad = False
            
    def forward(self, batch):
        ## input features: cdr3_tokens, chain_token_mask, cdr12_alpha_feat, cdr12_beta_feat, peptide, mhc, mhc_pseudo_mask
        _, tcr_single_repr = self.tcr_model(batch['cdr3_token'], batch['chain_token_mask'], batch['cdr12_alpha_feat'], batch['cdr12_beta_feat'], return_repr=True)
        pmhc_single_repr, pmhc_pairwise_repr, pmhc_attention_mask = self.pmhc_model(batch['peptide_token'], batch['mhc_embedding'], batch['mhc_pseudo_mask'], return_repr=True)
        tcr_attention_mask = batch['cdr3_token'].ne(self.tcr_model.padding_idx).int()
        
        if self.use_pmhc_struc_feat:
            pmhc_struc_repr = self.pmhc_struc_cnn(batch['pmhc_struc_feat'])
            pmhc_pairwise_repr = pmhc_pairwise_repr.clone() + pmhc_struc_repr.permute(0, 2, 3, 1)
        
        tcr_pairwise_repr = self.pairwise_tcr(tcr_single_repr.clone(), tcr_single_repr.clone())
        
        tcr2pmhc_pairwise_repr = self.pairwise_tcr2pmhc(tcr_single_repr.clone(), pmhc_single_repr.clone())
        pmhc2tcr_pairwise_repr = self.pairwise_pmhc2tcr(pmhc_single_repr.clone(), tcr_single_repr.clone())
        
        # concatenate tcr-pmhc representations and attention masks
        single_repr = torch.cat([tcr_single_repr, pmhc_single_repr], dim=1)
        attention_mask = torch.cat([tcr_attention_mask, pmhc_attention_mask], dim=1)
        
        # initialize pairwise repr of tcr pmhc
        pairwise_repr = torch.cat([
            torch.cat([tcr_pairwise_repr, tcr2pmhc_pairwise_repr], dim=2),
            torch.cat([pmhc2tcr_pairwise_repr, pmhc_pairwise_repr], dim=2)
        ], dim=1)
        
        # tcr pmhc pairformer modeling
        binding_single_repr, _ = self.binding_pairformer(
            single_repr = single_repr,
            pairwise_repr = pairwise_repr,
            mask = attention_mask
        )
        
        # layernorm and residual connection
        tcr_single_repr += self.tcr_layer_norm(self.tcr_self_attn(
            binding_single_repr[:, :tcr_single_repr.size(1), :], mask = tcr_attention_mask
        ))
        pmhc_single_repr += self.pmhc_layer_norm(self.pmhc_self_attn(
            binding_single_repr[:, tcr_single_repr.size(1):, :], mask = pmhc_attention_mask
        ))
        
        tcr_projection = tcr_single_repr[:, 0, :]
        pmhc_projection = pmhc_single_repr[:, 0, :]
        
        embed = torch.cat([tcr_projection, pmhc_projection], dim=1) 
        logits = self.binding_head(embed)
        
        tcr_projection = self.tcr_projector(tcr_projection)
        pmhc_projection = self.pmhc_projector(pmhc_projection)
        dist = F.cosine_similarity(tcr_projection, pmhc_projection, dim=-1)
    
        output = {'logits': logits, 'dist': dist, 'projection': (tcr_projection, pmhc_projection)}
            
        return output
    