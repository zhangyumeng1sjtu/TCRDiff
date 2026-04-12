import math
from tqdm import tqdm

import torch
import torch.nn as nn
import numpy as np
import torch.nn.functional as F

from .model_utils import stochastic_sample_from_categorical, topk_masking
from .model import TCRLM, TCRDPLM, PeptideMHCPairFormer, PairwiseCNN, PairFormerStack, PairwiseEmbedding, LMHead, Attention, TransformerLayerWithAdaLN, PeptideMHCModel


class ConditionalEncoder(nn.Module):
    def __init__(
        self,
        pmhc_model_config,
        dim_pairwise = 64,
        dropout = 0.1,
        use_pmhc_struc_feat = False,
        num_pmhc_struc_layers = 2,
    ):
        super(ConditionalEncoder, self).__init__()
        self.pmhc_model = PeptideMHCPairFormer(**pmhc_model_config)
        # self.pmhc_model = PeptideMHCModel(**pmhc_model_config)
        
        self.use_pmhc_struc_feat = use_pmhc_struc_feat
        if self.use_pmhc_struc_feat:
            self.pmhc_struc_cnn = PairwiseCNN(num_layers=num_pmhc_struc_layers, in_dim=3, embed_dim=dim_pairwise, kernel_size=1, dropout=dropout)
    
    def load_pretrained_weights(self, model_path):
        
        state_dict = torch.load(model_path, map_location='cpu', weights_only=True)
        
        pmhc_state_dict = {
            key.replace('pmhc_model.', ''): value for key, value in state_dict.items() if key.startswith('pmhc_model')
        }
        self.pmhc_model.load_state_dict(pmhc_state_dict)
        
        if self.use_pmhc_struc_feat:
            struc_state_dict = {
                key.replace('pmhc_struc_cnn.', ''): value for key, value in state_dict.items() if key.startswith('pmhc_struc_cnn')
            }
            self.pmhc_struc_cnn.load_state_dict(struc_state_dict)
        
        # freeze all encoder parameters
        for param in self.pmhc_model.parameters():
            param.requires_grad = False

        # unfreeze pmhc binding pairformer
        for param in self.pmhc_model.pmhc_binding_pairformer.parameters():
            param.requires_grad = True
        
    def forward(self, batch):
        pmhc_single_repr, pmhc_pairwise_repr, pmhc_attention_mask = self.pmhc_model(batch['peptide_token'], batch['mhc_embedding'], batch['mhc_pseudo_mask'], return_repr=True)
        
        if self.use_pmhc_struc_feat:
            pmhc_struc_repr = self.pmhc_struc_cnn(batch['pmhc_struc_feat'])
            pmhc_pairwise_repr = pmhc_pairwise_repr.clone() + pmhc_struc_repr.permute(0, 2, 3, 1)
        
        if 'proteinmpnn_alpha_logits' in batch and 'proteinmpnn_beta_logits' in batch:
            return {
                'pmhc_single_repr': pmhc_single_repr,
                'pmhc_pairwise_repr': pmhc_pairwise_repr,
                'pmhc_attention_mask': pmhc_attention_mask,
                'proteinmpnn_alpha_logits': batch['proteinmpnn_alpha_logits'],
                'proteinmpnn_beta_logits': batch['proteinmpnn_beta_logits'],
            }
        else:
            return {
                'pmhc_single_repr': pmhc_single_repr,
                'pmhc_pairwise_repr': pmhc_pairwise_repr,
                'pmhc_attention_mask': pmhc_attention_mask,
            }
        
        # for pmhc_model with no pairwise representation
        # pmhc_single_repr, pmhc_attention_mask = self.pmhc_model(batch['peptide_token'], batch['mhc_embedding'], batch['mhc_pseudo_mask'], return_repr=True) 
        # return {
        #     'pmhc_single_repr': pmhc_single_repr,
        #     'pmhc_attention_mask': pmhc_attention_mask,
        # }


class ConditionalAdapter(nn.Module):
    
    def __init__(
        self,
        dim_single = 256,
        dim_pairwise = 64,
        pair_bias_attn_dim_head = 32,
        pair_bias_attn_heads = 8,
        tri_attn_dim_head = 16,
        tri_attn_heads = 4,
        dropout = 0.1,
        pair_method = 'attention',
        **kwargs
    ):  
        super(ConditionalAdapter, self).__init__()
        
        self.pairwise_tcr = PairwiseEmbedding(dim_single, dim_pairwise, pair_method)
        self.pairwise_tcr2pmhc = PairwiseEmbedding(dim_single, dim_pairwise, pair_method)
        self.pairwise_pmhc2tcr = PairwiseEmbedding(dim_single, dim_pairwise, pair_method)
        self.dim_pairwise = dim_pairwise
        
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
        
        self.pmhc_layer_norm = nn.LayerNorm(dim_single)
        self.tcr_layer_norm = nn.LayerNorm(dim_single)
        
        self.tcr_cross_attn = Attention(
            dim=dim_single,
            dim_head=pair_bias_attn_dim_head,
            heads=pair_bias_attn_heads,
            dropout=dropout,
            gate_output=True, # False
            use_rotary_embedding=False
        )
        
        # self.dit_block = nn.ModuleList(
        #     [
        #         TransformerLayerWithAdaLN(
        #             dim_single,
        #             pair_bias_attn_dim_head,
        #             pair_bias_attn_heads,
        #             2 * dim_single,
        #             dropout = dropout
        #         )
        #         for _ in range(1) # set 2 DiT layers at first or 1 layer?
        #     ]
        # )

    def forward(self, tcr_single_repr, tcr_attention_mask, encoder_out):
        
        # load encoder outputs from a pMHC model
        pmhc_single_repr, pmhc_pairwise_repr, pmhc_attention_mask = encoder_out['pmhc_single_repr'], encoder_out['pmhc_pairwise_repr'], encoder_out['pmhc_attention_mask']
        
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
        
        # remove layer norm from binding_single_repr?
        tcr_binding_repr = self.tcr_layer_norm(binding_single_repr[:, :tcr_single_repr.size(1), :])
        pmhc_binding_repr = self.pmhc_layer_norm(binding_single_repr[:, tcr_single_repr.size(1):, :])
        
        # tcr_binding_repr = binding_single_repr[:, :tcr_single_repr.size(1), :]
        # pmhc_binding_repr = binding_single_repr[:, tcr_single_repr.size(1):, :]
        
        # add cross-attention between tcr and pmhc
        cross_attn_output = self.tcr_cross_attn(
            tcr_binding_repr, context=pmhc_binding_repr,
            mask=pmhc_attention_mask
        )
        # tcr_single_repr = cross_attn_output
        tcr_single_repr += cross_attn_output # remove residual connection??
                    
        return tcr_single_repr
    
        # add DiT blocks instead of cross-attention
        # pmhc_cls_repr = pmhc_binding_repr[:, 0, :]
        # for layer in self.dit_block:
        #     tcr_binding_repr = layer(tcr_binding_repr, pmhc_cls_repr, mask = tcr_attention_mask)
            
        # return tcr_binding_repr


# adapter for coembedding model
class ConditionalAdapter2(nn.Module):
    
    def __init__(
        self,
        dim = 256,
        self_attn_dim_head = 32,
        self_attn_heads = 8,
        dropout = 0.1,
        **kwargs
    ):  
        super(ConditionalAdapter2, self).__init__()
        
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
        
        self.dit_block = nn.ModuleList(
            [
                TransformerLayerWithAdaLN(
                    dim,
                    self_attn_dim_head,
                    self_attn_heads,
                    2 * dim,
                    dropout = dropout
                )
                for _ in range(2) # set 2 DiT layers at first or 1 layer?
            ]
        )

    def forward(self, tcr_single_repr, tcr_attention_mask, encoder_out):
        
        # load encoder outputs from a pMHC model
        pmhc_single_repr, pmhc_attention_mask = encoder_out['pmhc_single_repr'], encoder_out['pmhc_attention_mask']
       
        tcr_single_repr += self.tcr_layer_norm(self.tcr_self_attn(
            tcr_single_repr.clone(), mask = tcr_attention_mask
        ))
        pmhc_single_repr += self.pmhc_layer_norm(self.pmhc_self_attn(
            pmhc_single_repr.clone(), mask = pmhc_attention_mask
        ))
    
        # add DiT blocks instead of cross-attention
        pmhc_cls_repr = pmhc_single_repr[:, 0, :]
        for layer in self.dit_block:
            tcr_single_repr = layer(tcr_single_repr, pmhc_cls_repr, mask = tcr_attention_mask)
            
        return tcr_single_repr


class TCRDPLMWithCondition(nn.Module):
    
    def __init__(
        self,
        tcr_model_config,
        adapter_config,
        num_diffusion_steps = 100,
    ):
        super(TCRDPLMWithCondition, self).__init__()
        
        self.tcr_model = TCRDPLM(tcr_model_config)

        self.num_diffusion_timesteps = num_diffusion_steps
        
        # inherit tokenizer and special tokens
        self.padding_idx = self.tcr_model.padding_idx
        self.sep_idx = self.tcr_model.sep_idx
        self.mask_idx = self.tcr_model.mask_idx
        self.bos_idx = self.tcr_model.bos_idx
        self.eos_idx = self.tcr_model.eos_idx
        
        self.adapter_layer = ConditionalAdapter(**adapter_config)
        # self.adapter_layer = ConditionalAdapter2(**adapter_config)
        
        self.lm_head = LMHead(
            embed_dim = self.tcr_model.net.dim,
            output_dim = self.tcr_model.net.alphabet_size,
            weight = self.tcr_model.net.embed_tokens.weight
        )

    def load_pretrained_weights(self, model_path):
        state_dict = torch.load(model_path, map_location='cpu', weights_only=True)
        
        tcr_model_state_dict = {
            key.replace('tcr_model.', '') : value for key, value in state_dict.items() if key.startswith('tcr_model')
        }
        self.tcr_model.load_state_dict(tcr_model_state_dict)
        
        # use weights from pairformer in state_dict to be the initial weights of adapter layer
        adapter_state_dict = {
            key: value for key, value in state_dict.items() if key.startswith('pairwise_tcr') or key.startswith('pairwise_tcr2pmhc') or key.startswith('pairwise_pmhc2tcr') or key.startswith('binding_pairformer')
        }
        # for coembedding model
        # adapter_state_dict = {
        #     key: value for key, value in state_dict.items() if key.startswith('tcr_self_attn') or key.startswith('tcr_layer_norm') or key.startswith('pmhc_self_attn') or key.startswith('pmhc_layer_norm')
        # }
        
        self.adapter_layer.load_state_dict(adapter_state_dict, strict=False)
        
        for param in self.tcr_model.parameters():
            param.requires_grad = False
            
        # unfreeze some of the layers of pre-trained tcr model 
        # for param in self.tcr_model.net.transformer[-1].parameters():
        for param in self.tcr_model.net.parameters():
            param.requires_grad = True
    
    def forward(self, cdr3_tokens, chain_token_mask, cdr12_alpha_feat, cdr12_beta_feat, encoder_out=None, **kwargs):
        # further apply uncond_logits to use classifier-free gudiance
        uncond_logits, tcr_single_repr = self.tcr_model(cdr3_tokens, chain_token_mask, cdr12_alpha_feat, cdr12_beta_feat, return_repr=True)
        tcr_attention_mask = cdr3_tokens.ne(self.padding_idx).int()
        
        if encoder_out is not None:
            tcr_single_repr = self.adapter_layer(tcr_single_repr.clone(), tcr_attention_mask, encoder_out)
            
        logits = self.lm_head(tcr_single_repr)
        
        # eta = 1.5
        # cfg_logits = uncond_logits + eta * (logits - uncond_logits)
        # return cfg_logits, tcr_single_repr
        
        return logits, tcr_single_repr
    
    def compute_loss(self, batch, weighting='constant', encoder_out=None, partial_masks=None):
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
        
        batch['prev_tokens'] = x_t 
        
        # add time step embedding
        logits, _ = self.forward(x_t, chain_token_mask, cdr12_alpha_feat, cdr12_beta_feat, encoder_out=encoder_out)
        
        num_timesteps = self.num_diffusion_timesteps
        weight = {
            "linear": (num_timesteps - (t - 1)),  # num_timesteps * (1 - (t-1)/num_timesteps)
            "constant": num_timesteps * torch.ones_like(t)
        }[weighting][:, None].float() / num_timesteps
        
        return logits, target, loss_mask, weight
        
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


class ConditionalDPLM(nn.Module):
    
    def __init__(
        self,
        tcr_model_config,
        pmhc_model_config,
        adapter_config,
        num_diffusion_steps = 100,
        dim_pairwise = 64,
        dropout = 0.1,
        use_pmhc_struc_feat = True,
    ):
        super(ConditionalDPLM, self).__init__()
        
        self.encoder = ConditionalEncoder(
            pmhc_model_config, dim_pairwise, dropout, use_pmhc_struc_feat
        )
        self.decoder = TCRDPLMWithCondition(
            tcr_model_config, adapter_config, num_diffusion_steps
        )
        self.padding_idx = self.decoder.padding_idx
        self.sep_idx = self.decoder.sep_idx
        self.mask_idx = self.decoder.mask_idx
        self.bos_idx = self.decoder.bos_idx
        self.eos_idx = self.decoder.eos_idx
    
    def load_pretrained_weights(self, model_path):
        # load separate encoder and decoder weights (not use pre-trained binding model weights)
        # self.encoder.load_pretrained_weights(pmhc_model_path)
        # self.decoder.load_pretrained_weights(tcr_model_path)
        
        # optional: load pre-trained binding model weights
        self.encoder.load_pretrained_weights(model_path)
        self.decoder.load_pretrained_weights(model_path)
            
    def forward(self, batch, weighting='linear', partial_masks=None):
        encoder_out = self.encoder(batch)
        
        encoder_out['pmhc_single_repr'] = encoder_out['pmhc_single_repr'].repeat(2,1,1).detach()
        encoder_out['pmhc_pairwise_repr'] = encoder_out['pmhc_pairwise_repr'].repeat(2,1,1,1).detach() # rm when using the coembedding model
        encoder_out['pmhc_attention_mask'] = encoder_out['pmhc_attention_mask'].repeat(2,1).detach()
        
        logits, target, loss_mask, weight = self.decoder.compute_loss(
            batch=batch,
            weighting=weighting,
            encoder_out=encoder_out,
            partial_masks=partial_masks
        )
        
        return logits, target, loss_mask, weight
    
    def forward_encoder(self, batch):
        return self.encoder(batch)
    
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

            output_tokens = tokens.masked_fill(output_mask, self.mask_idx)
            output_scores = torch.zeros_like(output_tokens, dtype=torch.float)

            return output_tokens, output_scores
        
        return initial_output_tokens, initial_output_scores
    
    def forward_decoder(self, prev_decoder_out, encoder_out, chain_token_mask, cdr12_alpha_feat, cdr12_beta_feat, partial_masks=None, sampling_strategy='gumbel_argmax'):
        
        output_tokens = prev_decoder_out['output_tokens'].clone()
        output_scores = prev_decoder_out['output_scores'].clone()
        step, max_step = prev_decoder_out['step'], prev_decoder_out['max_step']
        temperature = prev_decoder_out['temperature']
        history = prev_decoder_out['history']

        output_masks = self.get_non_special_sym_mask(output_tokens, partial_masks=partial_masks)

        logits, last_hidden_state = self.decoder(output_tokens, chain_token_mask, cdr12_alpha_feat, cdr12_beta_feat, encoder_out)

        # Combine with ProteinMPNN logits if available
        if 'proteinmpnn_alpha_logits' in encoder_out and 'proteinmpnn_beta_logits' in encoder_out:
            # Split logits into alpha and beta parts
            alpha_mask = (chain_token_mask == 1)
            beta_mask = (chain_token_mask == 2)
            
            # Combine logits with ProteinMPNN logits
            eta = 0.15 # can be modified

            # print(logits[alpha_mask][2, 3:23], encoder_out['proteinmpnn_alpha_logits'][2, :])
            # only operate on the 20 positions reprsenting the amino acid tokens
            new_alpha_logits = logits[alpha_mask].clone().detach()
            new_beta_logits = logits[beta_mask].clone().detach()
            
            new_alpha_logits[:, 3:23] = logits[alpha_mask][:, 3:23] + eta * encoder_out['proteinmpnn_alpha_logits'].repeat(logits.shape[0], 1)
            new_beta_logits[:, 3:23] = logits[beta_mask][:, 3:23] + eta * encoder_out['proteinmpnn_beta_logits'].repeat(logits.shape[0], 1)

            logits[alpha_mask] = new_alpha_logits
            logits[beta_mask] = new_beta_logits

        logits[..., self.mask_idx] = -math.inf
        logits[..., self.padding_idx] = -math.inf
        logits[..., self.bos_idx] = -math.inf
        logits[..., self.eos_idx] = -math.inf
        logits[..., self.sep_idx] = -math.inf

        if sampling_strategy == 'argmax':
            logits = F.log_softmax(logits, dim=-1)
            _scores, _tokens = logits.max(-1)
        elif sampling_strategy == 'gumbel_argmax':
            noise_scale = 1.0 # 1.5
            _tokens, _scores = stochastic_sample_from_categorical(logits, temperature=temperature, noise_scale=noise_scale)  # small temperature -> less randomness
        
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
                noise_scale = 1.0 # 1.5
                lowest_k_mask[to_be_resample] = topk_masking(_scores_for_topk[to_be_resample], cutoff_len[to_be_resample], stochastic=True, temp=noise_scale * rate)
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
    
    
    def generate(self, batch, 
                 max_iter=None, temperature=None, 
                 partial_masks=None,
                 sampling_strategy='gumbel_argmax',
                 use_draft_seq=False):
        
        max_iter = max_iter
        temperature = temperature

        # 0) encoding
        encoder_out = self.forward_encoder(batch)
        # 1) initialized from all mask tokens
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

        for step in range(max_iter):
            # 2.1: predict
            with torch.no_grad():
                decoder_out = self.forward_decoder(
                    prev_decoder_out=prev_decoder_out,
                    encoder_out=encoder_out,
                    chain_token_mask=batch['chain_token_mask'],
                    cdr12_alpha_feat=batch['cdr12_alpha_feat'],
                    cdr12_beta_feat=batch['cdr12_beta_feat'],
                    partial_masks=partial_masks,
                    sampling_strategy=sampling_strategy
                )

            output_tokens = decoder_out['output_tokens']
            output_scores = decoder_out['output_scores']

            # 2.2: re-mask skeptical parts of low confidence
            non_special_sym_mask = self.get_non_special_sym_mask(
                prev_decoder_out['output_tokens'], partial_masks=partial_masks
            )
            
            output_masks, result_tokens, result_scores = self._reparam_decoding(
                output_tokens=prev_decoder_out['output_tokens'].clone(),
                output_scores=prev_decoder_out['output_scores'].clone(),
                cur_tokens=output_tokens.clone(),
                cur_scores=output_scores.clone(),
                decoding_strategy='reparam-uncond-deterministic-linear',
                # decoding_strategy='reparam-uncond-deterministic-cosine', # it seems cosine works better for structural data
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
    