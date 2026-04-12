import pandas as pd
import os


df = pd.read_csv('data/tcrpmhc/STCRDab-22-test-data-rm-dup.csv')

df['CDR3A_len'] = df['CDR3A'].apply(len)
df['CDR3B_len'] = df['CDR3B'].apply(len)

res = df[['PDB', 'Peptide', 'MHC', 'MHCA', 'MHCB', 'Organism', 'TRAV', 'TRBV', 'CDR3A_len', 'CDR3B_len']]

for i in range(len(res)):
    pdb = res.iloc[i, 0]
    peptide = res.iloc[i, 1]
    mhc = res.iloc[i, 2]
    mhca = res.iloc[i, 3]
    mhcb = res.iloc[i, 4]
    organism = res.iloc[i, 5]
    trav = res.iloc[i, 6]
    trbv = res.iloc[i, 7]
    cdr3a_len = res.iloc[i, 8]
    cdr3b_len = res.iloc[i, 9]
    
    cmd = f'python scripts/generate/generate_pmhc_binding_tcr.py \
    --config logs/tcr-pmhc-cond-dplm-cross-attn-finetune-tcr-dplm-all-constant/config.yml \
    --num_seqs 1 \
    --alpha_seq_len {cdr3a_len} \
    --beta_seq_len {cdr3b_len} \
    --peptide {peptide} \
    --mhc {mhc} \
    --mhca {mhca} \
    --mhcb {mhcb} \
    --trav {trav} \
    --trbv {trbv} \
    --organism {organism} \
    --temperature 0.1 \
    --sampling_strategy argmax \
    --max_iter 10 \
    --gpu_device 0 \
    --saveto gen.fasta/stcrdab_argmax_cosine/{pdb}'
    
    # cmd = f'python generate_pmhc_binding_tcr.py \
    # --config logs/tcr-pmhc-cond-dplm-cross-attn-finetune-tcr-dplm-all-constant/config.yml \
    # --num_seqs 1 \
    # --alpha_seq_len {cdr3a_len} \
    # --beta_seq_len {cdr3b_len} \
    # --peptide {peptide} \
    # --mhc {mhc} \
    # --mhca {mhca} \
    # --mhcb {mhcb} \
    # --trav {trav} \
    # --trbv {trbv} \
    # --organism {organism} \
    # --temperature 0.1 \
    # --sampling_strategy argmax \
    # --max_iter 10 \
    # --gpu_device 1 \
    # --saveto gen.fasta/stcrdab_argmax_cosine_proteimpnn_0.15/{pdb} \
    # --pdb {pdb}'
    
    print(cmd)
    os.system(cmd)
    