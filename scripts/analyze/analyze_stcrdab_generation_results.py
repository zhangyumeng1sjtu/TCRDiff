import argparse

import pandas as pd
import numpy as np

import glob
from Bio import SeqIO
from Bio import pairwise2
from Bio.Align import substitution_matrices

blosum62 = substitution_matrices.load("BLOSUM62")
gap_open = -5
gap_extend = -0.5
        
parser = argparse.ArgumentParser()
parser.add_argument('--test_data_path', type=str, default='data/tcrpmhc/STCRDab-22-test-data-rm-dup.csv')
parser.add_argument('--gen_data_dir', type=str, required=True)
parser.add_argument('--output_data_path', type=str, required=True)
parser.add_argument('--mode', type=str, default='max') # max or mean
args = parser.parse_args()

df = pd.read_csv(args.test_data_path)
pdb_ids = []

cdr3a_diversities = []
cdr3a_identities = []
cdr3a_blosum_scores = []

cdr3b_diversities = []
cdr3b_identities = []
cdr3b_blosum_scores = []

for pdb_id in df['PDB'].unique():
    pdb_ids.append(pdb_id)
    subset = df[df['PDB'] == pdb_id]

    cdr3a_preds = []
    cdr3b_preds = []

    fasta_files = glob.glob(f'{args.gen_data_dir}/{pdb_id}/*fasta')
    for file in fasta_files:
        for record in SeqIO.parse(file, 'fasta'):
            sequence = str(record.seq)
            cdr3a_preds.append(sequence.split('|')[0])
            cdr3b_preds.append(sequence.split('|')[1])

    # diversity: unique cdr3 sequences
    cdr3a_diversities.append(len(set(cdr3a_preds)) / len(cdr3a_preds))
    cdr3b_diversities.append(len(set(cdr3b_preds)) / len(cdr3b_preds))

    # compare with the original data
    cdr3a = subset['CDR3A'].values
    cdr3b = subset['CDR3B'].values

    cdr3a_max_identity, cdr3a_max_blosum_score = [], []
    for cdr3a_pred in cdr3a_preds:
        # Compute max sequence identity with any reference CDR3A
        # Using sequence alignment to compute identity
        max_identity, max_blosum_score = 0, 0
        for ref in cdr3a:
            # Use global alignment with BLOSUM62 scoring matrix
            alignments = pairwise2.align.globalds(cdr3a_pred, ref, blosum62, gap_open, gap_extend)
            if alignments:
                best_alignment = alignments[0]
                # Get the raw BLOSUM score from alignment
                max_blosum_score = max(max_blosum_score, best_alignment.score / (best_alignment.end - best_alignment.start))
        
            # Calculate character-level accuracy
            matches = sum(c1 == c2 for c1, c2 in zip(cdr3a_pred, ref))
            identity = matches / len(ref)
            max_identity = max(max_identity, identity)
        
        cdr3a_max_identity.append(max_identity)
        cdr3a_max_blosum_score.append(max_blosum_score)
        
    cdr3b_max_identity, cdr3b_max_blosum_score = [], []
    for cdr3b_pred in cdr3b_preds:
        # Compute max sequence identity with any reference CDR3B
        max_identity, max_blosum_score = 0, 0
        for ref in cdr3b:
            # Use global alignment with BLOSUM62 scoring matrix
            alignments = pairwise2.align.globalds(cdr3b_pred, ref, blosum62, gap_open, gap_extend)
            if alignments:
                best_alignment = alignments[0]
                # Get the raw BLOSUM score from alignment
                max_blosum_score = max(max_blosum_score, best_alignment.score / (best_alignment.end - best_alignment.start))
        
            # Calculate character-level accuracy
            matches = sum(c1 == c2 for c1, c2 in zip(cdr3b_pred, ref))
            identity = matches / len(ref)
            max_identity = max(max_identity, identity)
                
        cdr3b_max_identity.append(max_identity)
        cdr3b_max_blosum_score.append(max_blosum_score)
        
    if args.mode == 'max':
        cdr3a_identities.append(np.max(cdr3a_max_identity))
        cdr3b_identities.append(np.max(cdr3b_max_identity))
        cdr3a_blosum_scores.append(np.max(cdr3a_max_blosum_score))
        cdr3b_blosum_scores.append(np.max(cdr3b_max_blosum_score))
    elif args.mode == 'mean':
        cdr3a_identities.append(np.mean(cdr3a_max_identity))
        cdr3b_identities.append(np.mean(cdr3b_max_identity))
        cdr3a_blosum_scores.append(np.mean(cdr3a_max_blosum_score))
        cdr3b_blosum_scores.append(np.mean(cdr3b_max_blosum_score))
    
df_summary = pd.DataFrame({
    'PDB': pdb_ids,
    'CDR3A Diversity': cdr3a_diversities,
    'CDR3A Similarity': cdr3a_identities,
    'CDR3A Blosum Score': cdr3a_blosum_scores,
    'CDR3B Diversity': cdr3b_diversities,
    'CDR3B Similarity': cdr3b_identities,
    'CDR3B Blosum Score': cdr3b_blosum_scores
})
df_summary.to_csv(args.output_data_path, index=False)
