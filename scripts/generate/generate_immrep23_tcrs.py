import pandas as pd
import os

# generate without remask

df = pd.read_csv('data/tcrpmhc/immrep23_test_pos_data.csv')
df['CDR3A_len'] = df['CDR3A'].apply(len)
df['CDR3B_len'] = df['CDR3B'].apply(len)

res = df[['Peptide', 'MHC', 'MHCA', 'MHCB', 'Organism', 'TRAV', 'TRBV', 'CDR3A_len', 'CDR3B_len']].value_counts().reset_index()

print(res)

for i in range(len(res)):
    peptide = res.iloc[i, 0]
    mhc = res.iloc[i, 1]
    mhca = res.iloc[i, 2]
    mhcb = res.iloc[i, 3]
    organism = res.iloc[i, 4]
    trav = res.iloc[i, 5]
    trbv = res.iloc[i, 6]
    cdr3a_len = res.iloc[i, 7]
    cdr3b_len = res.iloc[i, 8]
    count = res.iloc[i, -1]
    
    cmd = f'python scripts/generate/generate_pmhc_binding_tcr.py \
    --config logs/tcr-pmhc-cond-dplm-cross-attn-finetune-tcr-dplm-all-constant/config.yml \
    --num_seqs {count * 10} \
    --alpha_seq_len {cdr3a_len} \
    --beta_seq_len {cdr3b_len} \
    --peptide {peptide} \
    --mhc {mhc} \
    --mhca {mhca} \
    --mhcb b2m \
    --trav {trav} \
    --trbv {trbv} \
    --organism human \
    --temperature 0.1 \
    --sampling_strategy gumbel_argmax \
    --max_iter 10 \
    --gpu_device 1 \
    --saveto gen.fasta/immrep23_01_cosine/{peptide}/{i}'

    print(peptide, mhc, trav, trbv, cdr3a_len, cdr3b_len, count)
    
    # print(cmd)
    os.system(cmd)
    