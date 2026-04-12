import pandas as pd
import glob
from Bio import SeqIO
from Bio import pairwise2
import json
import numpy as np

parser = argparse.ArgumentParser()
parser.add_argument('--test_data_path', type=str, default='data/tcrpmhc/STCRDab-22-test-data-rm-dup.csv')
parser.add_argument('--gen_data_dir', type=str, required=True)
parser.add_argument('--output_data_path', type=str, required=True)
args = parser.parse_args()

df = pd.read_csv(args.test_data_path)
df = df[~df['PDB'].isin(['7rm4', '8vcx', '8vcy', '8vd2'])]

data = []
max_aar = []
best_generations = []

max_cdr3a_aar = []
max_cdr3b_aar = []

def compute_amino_acid_recovery(ref, pred):
    # Compute the recovery of amino acids in the predicted sequence compared to the reference
    recovery = 0
    for i in range(len(ref)):
        if ref[i] == pred[i]:
            recovery += 1
    return recovery / len(ref)

for i in range(len(df)):
    pdb_id = df.iloc[i, 0]
    peptide = df.iloc[i, 1]
    mhc = df.iloc[i, 2]
    
    cdr3_alpha = df.iloc[i, 9]
    cdr3_beta = df.iloc[i, 14]
    
    gen_fasta = glob.glob(f'{args.gen_data_dir}/{pdb_id}/*fasta')
    if len(gen_fasta) == 0:
        print(f'No generated fasta file for {pdb_id}')
        continue
    gen_fasta = gen_fasta[0]
    
    generations = []
    for record in SeqIO.parse(gen_fasta, 'fasta'):
        sequence = str(record.seq)
        cdr3a_pred = sequence.split('|')[0]
        cdr3b_pred = sequence.split('|')[1]
        
        generations.append({
            'cdr3a': cdr3a_pred,
            'cdr3b': cdr3b_pred,
            'cdr3a_aar': compute_amino_acid_recovery(cdr3_alpha, cdr3a_pred),
            'cdr3b_aar': compute_amino_acid_recovery(cdr3_beta, cdr3b_pred),
            'aar': compute_amino_acid_recovery(cdr3_alpha + cdr3_beta, cdr3a_pred + cdr3b_pred)
        })

    data.append({
        'pdb_id': pdb_id,
        'peptide': peptide,
        'mhc': mhc,
        'cdr3a': cdr3_alpha,
        'cdr3b': cdr3_beta,
        'generations': generations,
    })
    
    # Find the generation with the best AAR
    # best_gen_idx = np.argmax([gen['aar'] for gen in generations])
    # best_gen = generations[best_gen_idx]
    
    # best_generations.append({
    #     'pdb_id': pdb_id,
    #     'peptide': peptide,
    #     'mhc': mhc,
    #     'cdr3a': cdr3_alpha,
    #     'cdr3b': cdr3_beta,
    #     'best_generation': best_gen
    # })
    
    max_aar.append(np.max([gen['aar'] for gen in generations]))
    max_cdr3a_aar.append(np.max([gen['cdr3a_aar'] for gen in generations]))
    max_cdr3b_aar.append(np.max([gen['cdr3b_aar'] for gen in generations]))
    
with open(args.output_data_path, 'w') as f:
    json.dump(data, f, indent=4)
    
print(np.mean(max_aar))
print(np.mean(max_cdr3a_aar))
print(np.mean(max_cdr3b_aar))
