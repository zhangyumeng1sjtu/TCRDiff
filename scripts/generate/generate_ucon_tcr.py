from argparse import ArgumentParser
import os
import numpy as np
import pandas as pd
import torch

from TCRDiff.model import TCRDPLM
from TCRDiff.utils import set_seed, load_config
from TCRDiff.utils.encoding import paired_cdr3_batch_encoding, paired_cdr12_batch_encoding


def extract_cdr12_seq(v_gene, organism, v_gene_library):
    if isinstance(v_gene, str):
        if v_gene == '':
            return np.nan, np.nan

        if len(v_gene.split('*')) == 1:
            v_gene = v_gene + '*01'

        gene_df = v_gene_library['human']
        cdrs = gene_df.loc[gene_df['id'] == v_gene, 'cdrs'].values[0]
        cdr1, cdr2 = cdrs.split(';')[0], cdrs.split(';')[1]
        return cdr1.replace('.', ''), cdr2.replace('.', '')
    else:
        return np.nan, np.nan


def get_initial(args, alpha_length, beta_length, tokenizer, device):
    alpha_seq = ['X'] * alpha_length
    beta_seq = ['X'] * beta_length
    if args.cond_chain_seq is not None:
        if args.cond_chain == 'alpha':
            assert alpha_length == len(args.cond_chain_seq)
            alpha_seq = args.cond_chain_seq
        elif args.cond_chain == 'beta':
            assert beta_length == len(args.cond_chain_seq)
            beta_seq = args.cond_chain_seq

    init_batch = [{
        'cdr1a': args.cdr1a,
        'cdr1b': args.cdr1b,
        'cdr2a': args.cdr2a,
        'cdr2b': args.cdr2b,
        'cdr3a': ''.join(alpha_seq),
        'cdr3b': ''.join(beta_seq),
    } for _ in range(args.num_seqs)]

    cdr3_tokens, chain_token_mask = paired_cdr3_batch_encoding(init_batch, tokenizer, 25)
    batch = {
        'cdr3_token': cdr3_tokens,
        'chain_token_mask': chain_token_mask,
    }
    cdr12_feat_alpha, cdr12_feat_beta = paired_cdr12_batch_encoding(init_batch)
    batch['cdr12_alpha_feat'] = cdr12_feat_alpha
    batch['cdr12_beta_feat'] = cdr12_feat_beta
    
    for key, value in batch.items():
        batch[key] = value.to(device)

    return batch


def generate(args):
    config_path = args.config
    config = load_config(config_path)
    set_seed(config.training.seed)
    device = torch.device(f"cuda:{config.training.gpu_device}" if torch.cuda.is_available() else "cpu")
    
    model = TCRDPLM(config.model, config.training.num_diffusion_steps, config.training.rdm_couple)
    tokenizer = model.net.tokenizer
    
    model.load_state_dict(torch.load(os.path.join(config.training.log_dir, 'checkpoint.pt'), weights_only=True))
    model = model.eval()
    model = model.to(device)

    max_iter = args.max_iter
    
    batch = get_initial(args, args.alpha_seq_len, args.beta_seq_len, tokenizer, device)
    partial_mask = batch['cdr3_token'].ne(model.mask_idx)
    
    with torch.amp.autocast('cuda'):
        outputs = model.generate(batch=batch, tokenizer=tokenizer,
                                    max_iter=max_iter,
                                    sampling_strategy=args.sampling_strategy,
                                    partial_masks=partial_mask)
    output_tokens = outputs[0]
    output_results = [''.join(seq.split(' ')) for seq in tokenizer.batch_decode(output_tokens, remove_special_tokens=True)]
    
    os.makedirs(args.saveto, exist_ok=True)
    saveto_name = os.path.join(args.saveto, f"iter_{max_iter}_L_{args.alpha_seq_len}_{args.beta_seq_len}.fasta")
    fp_save = open(saveto_name, 'w')
    for idx, seq in enumerate( 
        output_results
    ):
        fp_save.write(f">SEQUENCE_{idx}_L={args.alpha_seq_len}+{args.beta_seq_len}\n")
        fp_save.write(f"{seq}\n")
    fp_save.close()


if __name__ == '__main__':
    parser = ArgumentParser()

    parser.add_argument('--config', type=str, default='configs/config-pretrain-tcr-dplm.yml')
    parser.add_argument('--num_seqs', type=int, default=50)
    parser.add_argument('--alpha_seq_len', type=int)
    parser.add_argument('--beta_seq_len', type=int)
    
    parser.add_argument('--cond_chain', type=str) # alpha or beta
    parser.add_argument('--cond_chain_seq', type=str)
    
    parser.add_argument('--trav', type=str, default='')
    parser.add_argument('--trbv', type=str, default='')
    
    parser.add_argument('--saveto', type=str, default='gen.fasta')
    parser.add_argument('--temperature', type=float, default=1.0)
    parser.add_argument('--sampling_strategy', type=str, default='gumbel_argmax')
    parser.add_argument('--max_iter', type=int, default=100)
    
    args = parser.parse_args()

    v_genes = pd.read_csv('data/tcrpmhc/v_gene_lib.csv')
    v_gene_library = {'human': v_genes[v_genes['organism'] == 'human']}
    args.cdr1a, args.cdr2a = extract_cdr12_seq(args.trav, 'human', v_gene_library)
    args.cdr1b, args.cdr2b = extract_cdr12_seq(args.trbv, 'human', v_gene_library)

    generate(args)
