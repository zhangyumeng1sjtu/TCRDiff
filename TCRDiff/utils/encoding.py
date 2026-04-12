import torch
import numpy as np


atchley_factors = {
    'A': [-0.591, -1.302, -0.733, 1.570, -0.146],
    'C': [-1.343, 0.465, -0.862, -1.020, -0.255],
    'D': [1.050, 0.302, -3.656, -0.259, -3.242],
    'E': [1.357, -1.453, 1.477, 0.113, -0.837],
    'F': [-1.006, -0.590, 1.891, -0.397, 0.412],
    'G': [-0.384, 1.652, 1.330, 1.045, 2.064],
    'H': [0.336, -0.417, -1.673, -1.474, -0.078],
    'I': [-1.239, -0.547, 2.131, 0.393, 0.816],
    'K': [1.831, -0.561, 0.533, -0.277, 1.648],
    'L': [-1.019, -0.987, -1.505, 1.266, -0.912],
    'M': [-0.663, -1.524, 2.219, -1.005, 1.212],
    'N': [0.945, 0.828, 1.299, -0.169, 0.933],
    'P': [0.189, 2.081, -1.628, 0.421, -1.392],
    'Q': [0.931, -0.179, -3.005, -0.503, -1.853],
    'R': [1.538, -0.055, 1.502, 0.440, 2.897],
    'S': [-0.228, 1.399, -4.760, 0.670, -2.647],
    'T': [-0.032, 0.326, 2.213, 0.908, 1.313],
    'V': [-1.337, -0.279, -0.544, 1.242, -1.262],
    'W': [-0.595, 0.009, 0.672, -2.128, -0.184],
    'Y': [0.260, 0.830, 3.097, -0.838, 1.512],
    '*': [0.0, 0.0, 0.0, 0.0, 0.0]
}

BLOSUM50_MATRIX = np.array([
    [5, -2, -1, -2, -1, -3, 0, -2, -1, -2, -1, -1, -1, -2, -1, 1, 0, -3, -2, 0],
    [-2, 7, -1, -2, -4, 1, -1, -3, 0, -4, -3, 3, -2, -2, -3, -1, -1, -3, -1, -3],
    [-1, -1, 7, 2, -2, 0, 0, 0, 1, -3, -4, 0, -2, -4, -2, 1, 0, -4, -2, -3],
    [-2, -2, 2, 8, -4, 0, 2, -1, -1, -4, -4, -1, -4, -5, -1, 0, -1, -5, -3, -4],
    [-1, -4, -2, -4, 11, -3, -4, -3, -4, -5, -5, -3, -5, -4, -3, -1, -1, -4, -4, -1],
    [-3, 1, 0, 0, -3, 7, -1, -2, 1, -2, -1, 2, 0, -3, -2, 0, -1, -1, -1, -2],
    [0, -1, 0, 2, -4, -1, 7, -2, -1, -4, -3, 0, -2, -4, -1, 0, -1, -4, -3, -3],
    [-2, -3, 0, -1, -3, -2, -2, 5, -2, -3, -3, -2, -3, -3, -2, 0, -2, -3, -3, -4],
    [-1, 0, 1, -1, -4, 1, -1, -2, 5, -3, -2, 1, 0, -3, -1, 0, -1, -3, -2, -3],
    [-2, -4, -3, -4, -5, -2, -4, -3, -3, 5, 2, -3, 2, 0, -3, -3, -2, -2, 1, 4],
    [-1, -3, -4, -4, -5, -1, -3, -3, -2, 2, 5, -2, 3, 1, -4, -3, -2, -2, 0, 1],
    [-1, 3, 0, -1, -3, 2, 0, -2, 1, -3, -2, 6, -2, -4, -1, 0, -1, -3, -2, -3],
    [-1, -2, -2, -4, -5, 0, -2, -3, 0, 2, 3, -2, 7, 0, -4, -2, -1, -1, 1, 2],
    [-2, -2, -4, -5, -4, -3, -4, -3, -3, 0, 1, -4, 0, 8, -3, -2, -2, 4, 4, 0],
    [-1, -3, -2, -1, -3, -2, -1, -2, -1, -3, -4, -1, -4, -3, 9, -1, -1, -3, -3, -3],
    [1, -1, 1, 0, -1, 0, 0, 0, 0, -3, -3, 0, -2, -2, -1, 5, 2, -4, -2, -1],
    [0, -1, 0, -1, -1, -1, -1, -2, -1, -2, -2, -1, -1, -2, -1, 2, 5, -3, -2, 0],
    [-3, -3, -4, -5, -4, -1, -4, -3, -3, -2, -2, -3, -1, 4, -3, -4, -3, 15, 2, -1],
    [-2, -1, -2, -3, -4, -1, -3, -3, -2, 1, 0, -2, 1, 4, -3, -2, -2, 2, 8, 0],
    [0, -3, -3, -4, -1, -2, -3, -4, -3, 4, 1, -3, 2, 0, -3, -1, 0, -1, 0, 5]
])

max_cdr1a_len = 8
max_cdr2a_len = 8 
max_cdr1b_len = 6
max_cdr2b_len = 7


def peptide_batch_encoding(seq_str_list, tokenizer, batch_size, max_peptide_len=None):

    if max_peptide_len:
        seq_str_list = [seq_str[:max_peptide_len] for seq_str in seq_str_list]
        
    seq_encoded_list = [tokenizer.encode(seq_str) for seq_str in seq_str_list]
    
    max_len = max(len(seq_encoded) for seq_encoded in seq_encoded_list)
    tokens = torch.empty((batch_size, max_len + int(tokenizer.prepend_bos) + int(tokenizer.append_eos)), dtype=torch.long)
    tokens.fill_(tokenizer.padding_idx)

    strs = []
    for i, (seq_str, seq_encoded) in enumerate(zip(seq_str_list, seq_encoded_list)):
        strs.append(seq_str)

        if tokenizer.prepend_bos:
            tokens[i, 0] = tokenizer.cls_idx

        seq = torch.tensor(seq_encoded, dtype=torch.long)
        tokens[i, int(tokenizer.prepend_bos): len(seq_encoded) + int(tokenizer.prepend_bos),] = seq

        if tokenizer.append_eos:
            tokens[i, len(seq_encoded) + int(tokenizer.prepend_bos)] = tokenizer.eos_idx
            
    return strs, tokens
        

#TODO: MHC blosum50 encoding (classI + classII)
def blosum_encoding(seq, matrix):
    amino_acids = "ARNDCQEGHILKMFPSTWYV"
    amino_acid_to_index = {aa: i for i, aa in enumerate(amino_acids)}
    seq_len = len(seq)
    
    # Initialize an empty encoded matrix (-1 for padding/gap/unk positions)
    encoded_seq = np.full((seq_len, 20), -1)
    for i, aa in enumerate(seq):
        if aa in amino_acids:
            encoded_seq[i] = matrix[amino_acid_to_index[aa]]

    return encoded_seq

 
def pad_protein_sequence(sequence, max_seq_len, pad_token='X'):
    padded_sequence = sequence.ljust(max_seq_len, pad_token)
    return padded_sequence[:max_seq_len]


def mhc_allele_to_seq(allele, mhc_library):
    for key in mhc_library.keys():
        if key.startswith(allele):
            return mhc_library[key]
    return None


def mhc_encoding(mhc_alpha_seq, mhc_beta_seq=None, mhc_class='I', max_seq_len=175):
    # concat alpha and beta sequence for mhc class II alleles
    mhc_seq = mhc_alpha_seq + mhc_beta_seq if mhc_class == 'II' else mhc_alpha_seq
    seq = pad_protein_sequence(mhc_seq, max_seq_len)
    blosum50_feat = blosum_encoding(seq, BLOSUM50_MATRIX)

    return blosum50_feat


# cdr3 tokenization with nan input handling
def paired_cdr3_batch_encoding(batch, tokenizer, max_cdr3_len=None):
    batch_size = len(batch) # batch is a list of dict
    
    cdr3_encoded_list = []
    alpha_indices = []
    beta_indices = []
    
    # get encoded sequence and chain indices
    for data in batch:
        cdr3_alpha = data['cdr3a'] if isinstance(data['cdr3a'], str) else ''
        cdr3_beta = data['cdr3b'] if isinstance(data['cdr3b'], str) else ''
        
        # cdr3_alpha = data['cdr3a'] if isinstance(data['cdr3a'], str) else '<na>' # TODO: set NA to 'Z' and encode using a mask-chain token?
        # cdr3_beta = data['cdr3b'] if isinstance(data['cdr3b'], str) else '<na>'
        
        # cdr3_alpha = '<na>' if len(cdr3_alpha) == 0 else cdr3_alpha
        # cdr3_beta = '<na>' if len(cdr3_beta) == 0 else cdr3_beta

        if max_cdr3_len is not None:
            cdr3_alpha = cdr3_alpha[:max_cdr3_len] if cdr3_alpha else '' 
            cdr3_beta = cdr3_beta[:max_cdr3_len] if cdr3_beta else ''
        
        cdr3_encoded_list.append(tokenizer.encode(cdr3_alpha, cdr3_beta))
        alpha_indices.append([0, len(cdr3_alpha)])
        beta_indices.append([len(cdr3_alpha) + 1, len(cdr3_alpha) + len(cdr3_beta) + 1])

    # create the batch encoding
    max_len = max(len(cdr3_encoded) for cdr3_encoded in cdr3_encoded_list)
    tokens = torch.empty((batch_size, max_len + int(tokenizer.prepend_bos) + int(tokenizer.append_eos)), dtype=torch.long)
    tokens.fill_(tokenizer.padding_idx)
    chain_token_mask = torch.zeros((batch_size, max_len + int(tokenizer.prepend_bos) + int(tokenizer.append_eos)), dtype=torch.long)
    
    for i, seq_encoded in enumerate(cdr3_encoded_list):
        
        if tokenizer.prepend_bos:
            tokens[i, 0] = tokenizer.cls_idx
            
        seq = torch.tensor(seq_encoded, dtype=torch.long)
        tokens[i, int(tokenizer.prepend_bos): len(seq_encoded) + int(tokenizer.prepend_bos),] = seq
        chain_token_mask[i, alpha_indices[i][0] + int(tokenizer.prepend_bos): alpha_indices[i][1] + int(tokenizer.prepend_bos)] = 1
        chain_token_mask[i, beta_indices[i][0] + int(tokenizer.prepend_bos): beta_indices[i][1] + int(tokenizer.prepend_bos)] = 2
        
        if tokenizer.append_eos:
            tokens[i, len(seq_encoded) + int(tokenizer.prepend_bos)] = tokenizer.eos_idx
        
    return tokens, chain_token_mask


# cdr12 atchley factor encoding with nan input handling
def paired_cdr12_batch_encoding(batch):
    batch_size = len(batch)
    
    cdr1a_atchley_feat = torch.zeros(batch_size, max_cdr1a_len, 5)
    cdr2a_atchley_feat = torch.zeros(batch_size, max_cdr2a_len, 5)
    cdr1b_atchley_feat = torch.zeros(batch_size, max_cdr1b_len, 5)
    cdr2b_atchley_feat = torch.zeros(batch_size, max_cdr2b_len, 5)
    
    for i, data in enumerate(batch):
        cdr1_alpha = data['cdr1a'] if isinstance(data['cdr1a'], str) else ''
        cdr2_alpha = data['cdr2a'] if isinstance(data['cdr2a'], str) else ''
        
        cdr1_beta = data['cdr1b'] if isinstance(data['cdr1b'], str) else ''
        cdr2_beta = data['cdr2b'] if isinstance(data['cdr2b'], str) else ''
        
        if cdr1_alpha:
            cdr1a_atchley_feat[i, :len(cdr1_alpha), :] = torch.tensor([atchley_factors[aa] for aa in cdr1_alpha], dtype=torch.float)
        if cdr2_alpha:
            cdr2a_atchley_feat[i, :len(cdr2_alpha), :] = torch.tensor([atchley_factors[aa] for aa in cdr2_alpha], dtype=torch.float)
        if cdr1_beta:
            cdr1b_atchley_feat[i, :len(cdr1_beta), :] = torch.tensor([atchley_factors[aa] for aa in cdr1_beta], dtype=torch.float)
        if cdr2_beta:
            cdr2b_atchley_feat[i, :len(cdr2_beta), :] = torch.tensor([atchley_factors[aa] for aa in cdr2_beta], dtype=torch.float)
    
    cdr12_feat_alpha = torch.cat((cdr1a_atchley_feat, cdr2a_atchley_feat), dim=1)
    cdr12_feat_beta = torch.cat((cdr1b_atchley_feat, cdr2b_atchley_feat), dim=1)
    
    return cdr12_feat_alpha, cdr12_feat_beta


def pmhc_struc_feat_batch_encoding(batch, dim, feat_dict, max_dist_threshold=20, max_pae_threshold=5):
    
    batch_size = len(batch)
    pmhc_struc_feat = torch.zeros(batch_size, 3, dim, dim) # distogram, contact, pae
    
    for i, data in enumerate(batch):
        peptide, mhc = data['peptide'], data['mhc']
        feat = feat_dict['_'.join([peptide, mhc])]
        feat_shape = feat['distogram'].shape[0] 
        pmhc_struc_feat[i, 0, :feat_shape, :feat_shape] = torch.tensor(feat['distogram'] / max_dist_threshold, dtype=torch.float).clamp(0, 1)
        pmhc_struc_feat[i, 1, :feat_shape, :feat_shape] = torch.tensor(feat['contact'], dtype=torch.float)
        pmhc_struc_feat[i, 2, :feat_shape, :feat_shape] = torch.tensor(feat['pae'] / max_pae_threshold, dtype=torch.float).clamp(0, 1)
    
    return pmhc_struc_feat
