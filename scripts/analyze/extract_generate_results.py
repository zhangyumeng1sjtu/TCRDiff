import pandas as pd
import numpy as np

import glob
from Bio import SeqIO
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--test_data_path', type=str, default='data/tcrpmhc/immrep23_test_pos_data.csv')
parser.add_argument('--gen_data_dir', type=str, required=True)
parser.add_argument('--output_data_path', type=str, required=True)
args = parser.parse_args()

df = pd.read_csv(args.test_data_path)
df['CDR3A_len'] = df['CDR3A'].apply(len)
df['CDR3B_len'] = df['CDR3B'].apply(len)

res = df[['Peptide', 'MHC', 'MHCA', 'MHCB', 'Organism', 'TRAV', 'TRBV', 'CDR3A_len', 'CDR3B_len']].value_counts().reset_index()

data_list = []

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
    
    cdr3a_preds = []
    cdr3b_preds = []
    
    fasta_file = glob.glob(f'{args.gen_data_dir}/{peptide}/{i}/*fasta')[0]
    for record in SeqIO.parse(fasta_file, 'fasta'):
        sequence = str(record.seq)
        cdr3a_preds.append(sequence.split('|')[0])
        cdr3b_preds.append(sequence.split('|')[1])
        
    data = pd.DataFrame({'CDR3A': cdr3a_preds, 'CDR3B': cdr3b_preds})
    
    data['Peptide'] = peptide
    data['MHC'] = mhc
    data['MHCA'] = mhca
    data['MHCB'] = mhcb
    data['Organism'] = organism
    data['TRAV'] = trav
    data['TRBV'] = trbv
    
    data_list.append(data)
    
final_data = pd.concat(data_list)
final_data.to_csv(args.output_data_path, index=False)
