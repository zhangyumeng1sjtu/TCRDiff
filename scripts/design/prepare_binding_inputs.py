from Bio import SeqIO
import glob
import pandas as pd

DESIGN_PIPELINE_DIR='design_pipeline/MAGE-A3'

fasta_file = glob.glob(f'{DESIGN_PIPELINE_DIR}/*.fasta')[0]
cdr3a_gen = []
cdr3b_gen = []

for record in SeqIO.parse(fasta_file, 'fasta'):
    sequence = str(record.seq)
    cdr3a_gen.append(sequence.split('|')[0])
    cdr3b_gen.append(sequence.split('|')[1])

df = pd.DataFrame({'CDR3A': cdr3a_gen, 'CDR3B': cdr3b_gen})
df['Peptide'] = 'EVDPIGHLY'
df['MHC'] = 'HLA-A*01:01'
df['MHCA'] = 'A*01:01'
df['MHCB'] = 'b2m'
df['TRAV'] = 'TRAV21*01'
df['TRBV'] = 'TRBV5-1*01'
df['Organism'] = 'human'
df.to_csv(f'{DESIGN_PIPELINE_DIR}/MAGE-A3_cdr3_generation.csv', index=False)

off_target_peptides = ['ESDPIVAQY']
for peptide in off_target_peptides:
    df['Peptide'] = peptide
    df.to_csv(f'{DESIGN_PIPELINE_DIR}/MAGE-A3_cdr3_generation_{peptide}.csv', index=False)


# sample 10000 TCRs from background distribution
bg_df = pd.read_csv('data/tcrpmhc/healthy/human_tcr_filtered_2.csv')

# filtered_df = bg_df[bg_df['TRAV'].str.startswith('TRAV21') & bg_df['TRBV'].str.startswith('TRBV5-1')]
sampled_bg_df = bg_df.sample(n=10000, replace=True)
sampled_bg_df['Peptide'] = 'EVDPIGHLY'
sampled_bg_df['MHC'] = 'HLA-A*01:01'
sampled_bg_df['MHCA'] = 'A*01:01'
sampled_bg_df['MHCB'] = 'b2m'
sampled_bg_df['Organism'] = 'human'

sampled_bg_df.to_csv(f'{DESIGN_PIPELINE_DIR}/sampled_bg_df.csv', index=False)

for peptide in ['ESDPIVAQY']:
    sampled_bg_df['Peptide'] = peptide
    sampled_bg_df.to_csv(f'{DESIGN_PIPELINE_DIR}/sampled_bg_df_{peptide}.csv', index=False)
    