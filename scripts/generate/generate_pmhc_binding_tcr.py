from argparse import ArgumentParser
import os
import torch
import numpy as np
import pandas as pd
import pickle
import json
import torch.nn.functional as F

import sys
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from TCRDiff.trainer import ConditionalTCRDPLMTrainer
from TCRDiff.utils import set_seed, load_config
from TCRDiff.utils.encoding import paired_cdr3_batch_encoding, paired_cdr12_batch_encoding, mhc_allele_to_seq, peptide_batch_encoding, mhc_encoding, pmhc_struc_feat_batch_encoding
from TCRDiff.utils.tokenizer import PairCDR3Tokenizer, PeptideTokenizer


MH1_PSEUDO_POS = [6, 8, 23, 44, 58, 61, 62, 65, 66, 68, 69, 72, 73, 75, 76, 79, 80,
                  83, 94, 96, 98, 113, 115, 117, 142, 146, 149, 151, 155, 157, 158, 162, 166, 170]

MH2_PSEUDO_POS = [4, 7, 18, 20, 27, 48, 49, 54, 55, 57, 61, 62, 64, 68, 69, # alpha
                  77, 79, 81, 94, 96, 98, 115, 125, 135, 138, 139, 142, 145, 146, 149, 153, 154, 157, 158] # beta (75+x)


def transform_mhc_allele(mhc_allele, mhc_library):
        if mhc_allele == 'b2m':
            return None
        else:
            mhc_seq = mhc_allele_to_seq(mhc_allele.replace('HLA-', ''), mhc_library)
            assert mhc_seq is not None
            return mhc_seq


def extract_cdr12_seq(v_gene, organism, v_gene_library):
        if isinstance(v_gene, str):
            if v_gene == '':
                return np.nan, np.nan
            
            if len(v_gene.split('*')) == 1:
                v_gene = v_gene + '*01'
                
            gene_df = v_gene_library[organism]
            assert v_gene in gene_df['id'].values # 
            
            cdrs = gene_df[gene_df['id'] == v_gene]['cdrs'].values[0]
            cdr1, cdr2 = cdrs.split(';')[0], cdrs.split(';')[1]
            return cdr1.replace('.', ''), cdr2.replace('.', '')
        else:
            return np.nan, np.nan


def prepare_input_batch(args, alpha_length, beta_length, device, alpha_seq=None, beta_seq=None, proteinmpnn_alpha_logits=None, proteinmpnn_beta_logits=None):
    tcr_tokenizer = PairCDR3Tokenizer()
    pep_tokenizer = PeptideTokenizer()
    
    # optional: set partial alpha or beta sequence
    if alpha_seq is None:
        alpha_seq = ['X'] * alpha_length
    else:
        assert len(alpha_seq) == alpha_length
        alpha_seq = list(alpha_seq)

    # optional: set partial beta sequence
    if beta_seq is None:
        beta_seq = ['X'] * beta_length
    else:
        assert len(beta_seq) == beta_length
        beta_seq = list(beta_seq)

    # prepare v gene and mhc library
    v_gene_library = {}
    v_genes = pd.read_csv(args.v_gene_lib_path)    
    for org in ['human', 'mouse']:
        v_gene_library[org] = v_genes[v_genes['organism'] == org]
    with open(args.mhc_library, 'r') as f:
        mhc_library = json.load(f)
    
    cdr1a, cdr2a = extract_cdr12_seq(args.trav, args.organism, v_gene_library)
    cdr1b, cdr2b = extract_cdr12_seq(args.trbv, args.organism, v_gene_library)
    
    init_batch = [{
        'cdr3a': ''.join(alpha_seq),
        'cdr3b': ''.join(beta_seq),
        'cdr1a': cdr1a,
        'cdr2a': cdr2a,
        'cdr1b': cdr1b,
        'cdr2b': cdr2b,
        'peptide': args.peptide,
        'mhc': args.mhc,
        'mhca': transform_mhc_allele(args.mhca, mhc_library),
        'mhcb': transform_mhc_allele(args.mhcb, mhc_library),
        'class': 'I' if args.mhcb == 'b2m' else 'II',
        'organism': args.organism,
    } for _ in range(args.num_seqs)]
    
    # transform input tcr and pmhc features
    cdr3_tokens, chain_token_mask = paired_cdr3_batch_encoding(init_batch, tcr_tokenizer, 25)
    cdr12_feat_alpha, cdr12_feat_beta = paired_cdr12_batch_encoding(init_batch)
    
    # Create partial mask for standard amino acid tokens
    partial_mask = torch.zeros_like(cdr3_tokens, dtype=torch.bool)
    for idx in tcr_tokenizer.standard_token_idxes:
        partial_mask = partial_mask | (cdr3_tokens == idx)
    
    _, peptide_tokens = peptide_batch_encoding([d['peptide'] for d in init_batch], pep_tokenizer, args.num_seqs, 25)
    mhc_embs = torch.tensor(np.stack([mhc_encoding(d['mhca'], d['mhcb'], d['class'], 175) for d in init_batch], axis=0), dtype=torch.float)
    mhc_pseudo_mask = torch.zeros((args.num_seqs, 175), dtype=torch.long)
    for i, mhc_class in enumerate([d['class'] for d in init_batch]):
        if mhc_class == 'I':
            mhc_pseudo_mask[i, MH1_PSEUDO_POS] = 1
        elif mhc_class == 'II':
            mhc_pseudo_mask[i, MH2_PSEUDO_POS] = 1 
    
    batch = {
        'cdr3_token': cdr3_tokens,
        'chain_token_mask': chain_token_mask,
        'cdr12_alpha_feat': cdr12_feat_alpha,
        'cdr12_beta_feat': cdr12_feat_beta,
        'peptide_token': peptide_tokens,
        'mhc_embedding': mhc_embs,
        'mhc_pseudo_mask': mhc_pseudo_mask,
        'partial_mask': partial_mask
    }

    # add pmhc structural features
    if args.use_struc_feat:
        with open(args.pmhc_struc_feat_path, 'rb') as f:
            pmhc_struc_feat_dict = pickle.load(f)
        batch['pmhc_struc_feat'] = pmhc_struc_feat_batch_encoding(init_batch, dim=peptide_tokens.shape[1] + 34, feat_dict=pmhc_struc_feat_dict)
    
    # add proteinmpnn logits if provided
    if proteinmpnn_alpha_logits is not None:
        batch['proteinmpnn_alpha_logits'] = torch.tensor(proteinmpnn_alpha_logits, dtype=torch.float).to(device)
    if proteinmpnn_beta_logits is not None:
        batch['proteinmpnn_beta_logits'] = torch.tensor(proteinmpnn_beta_logits, dtype=torch.float).to(device)
    
    # move to device
    for key, value in batch.items():
        batch[key] = value.to(device)

    return batch


def generate(args, proteinmpnn_alpha_logits=None, proteinmpnn_beta_logits=None):
    config_path = args.config
    config = load_config(config_path)
    set_seed(config.training.seed)
    
    device = torch.device(f"cuda:{args.gpu_device}" if torch.cuda.is_available() else "cpu")
    config.training.gpu_device = args.gpu_device
    
    Trainer = ConditionalTCRDPLMTrainer(config)
    Trainer.model.load_state_dict(torch.load(os.path.join(config.training.log_dir, 'checkpoint.pt'), weights_only=True))
    
    Trainer.model.eval()
    max_iter = args.max_iter
    
    # Split batch if num_seqs is larger than 100
    batch_size = 100
    all_output_results = []
    
    for batch_start in range(0, args.num_seqs, batch_size):
        batch_end = min(batch_start + batch_size, args.num_seqs)
        current_num_seqs = batch_end - batch_start
        
        # Create a temporary args object with the current batch size
        temp_args = type('Args', (), {})()
        for attr in dir(args):
            if not attr.startswith('_'):
                setattr(temp_args, attr, getattr(args, attr))
        temp_args.num_seqs = current_num_seqs
        
        batch = prepare_input_batch(temp_args, args.alpha_seq_len, args.beta_seq_len, device, alpha_seq=args.alpha_seq, beta_seq=args.beta_seq, 
                                  proteinmpnn_alpha_logits=proteinmpnn_alpha_logits, proteinmpnn_beta_logits=proteinmpnn_beta_logits)
        
        outputs = Trainer.forward(batch, max_iter, args.sampling_strategy, args.temperature, partial_masks=batch['partial_mask'])
        output_tokens, output_scores = outputs
        
        output_results = [''.join(seq.split(' ')) for seq in Trainer.tokenizer.batch_decode(output_tokens, remove_special_tokens=True)]
        all_output_results.extend(output_results)
    
    os.makedirs(args.saveto, exist_ok=True)
    saveto_name = os.path.join(args.saveto, f"{args.peptide}_{args.mhc.replace('/','-')}_{args.trav.replace('/','-')}_{args.trbv.replace('/','-')}_iter_{max_iter}_L_{args.alpha_seq_len}_{args.beta_seq_len}.fasta")
    
    fp_save = open(saveto_name, 'w')
    for idx, seq in enumerate(all_output_results):
        fp_save.write(f">SEQUENCE_{idx}\n")
        fp_save.write(f"{seq}\n")
    fp_save.close()


if __name__ == '__main__':
    parser = ArgumentParser()

    parser.add_argument('--config', type=str, default='configs/config-train-tcr-pmhc-dplm.yml')
    parser.add_argument('--num_seqs', type=int, default=100)
    parser.add_argument('--alpha_seq_len', type=int, default=15)
    parser.add_argument('--beta_seq_len', type=int, default=15)
    parser.add_argument('--alpha_seq', type=str, default=None)
    parser.add_argument('--beta_seq', type=str, default=None)

    parser.add_argument('--peptide', type=str, required=True)
    parser.add_argument('--mhc', type=str, required=True)
    parser.add_argument('--mhca', type=str, required=True)
    parser.add_argument('--mhcb', type=str, required=True)
    parser.add_argument('--trav', type=str, default='')
    parser.add_argument('--trbv', type=str, default='')
    parser.add_argument('--organism', type=str, default='human')
    
    parser.add_argument('--mhc_library', type=str, default='data/pmhc/mhc_lib.json')
    parser.add_argument('--v_gene_lib_path', type=str, default='data/tcrpmhc/v_gene_lib.csv')
    
    parser.add_argument('--use_struc_feat', action='store_true')
    parser.add_argument('--pmhc_struc_feat_path', type=str, default='data/tcrpmhc/pmhc_structural_data.pkl')
    
    parser.add_argument('--saveto', type=str, default='gen.fasta')
    parser.add_argument('--temperature', type=float, default=1.0)
    parser.add_argument('--sampling_strategy', type=str, default='gumbel_argmax')
    parser.add_argument('--max_iter', type=int, default=10)
    
    parser.add_argument('--gpu_device', type=int, default=0)
    parser.add_argument('--pdb', type=str, default='')
    
    args = parser.parse_args()

    if args.pdb != '':
        with open('data/stcrdab_test_proteimpnn_cdr3_logits.pkl', 'rb') as f:
            proteinmpnn_cdr3_logits = pickle.load(f)

        proteinmpnn_cdr3a_logits = proteinmpnn_cdr3_logits[args.pdb]['cdr3a_logits'][:, :20]
        proteinmpnn_cdr3b_logits = proteinmpnn_cdr3_logits[args.pdb]['cdr3b_logits'][:, :20]

        generate(args, proteinmpnn_cdr3a_logits, proteinmpnn_cdr3b_logits)

    else:
        generate(args)
