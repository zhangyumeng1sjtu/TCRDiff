from typing import Sequence
import numpy as np
import pandas as pd
import torch
import random
import pickle

from ..utils.tokenizer import PeptideTokenizer, PairCDR3Tokenizer
from ..utils.encoding import peptide_batch_encoding, paired_cdr3_batch_encoding, paired_cdr12_batch_encoding, mhc_encoding, pmhc_struc_feat_batch_encoding


MH1_PSEUDO_POS = [6, 8, 23, 44, 58, 61, 62, 65, 66, 68, 69, 72, 73, 75, 76, 79, 80,
                  83, 94, 96, 98, 113, 115, 117, 142, 146, 149, 151, 155, 157, 158, 162, 166, 170]

MH2_PSEUDO_POS = [4, 7, 18, 20, 27, 48, 49, 54, 55, 57, 61, 62, 64, 68, 69, # alpha
                  77, 79, 81, 94, 96, 98, 115, 125, 135, 138, 139, 142, 145, 146, 149, 153, 154, 157, 158] # beta (75+x)

# entropy values for each token position in CDR3α and CDR3β (normalized)
CDR3A_ENTROPY = [0.50387849, 0.84236551, 1.57868472, 1.86885993, 2.        ,
                1.93329058, 1.79237105, 1.77259808, 1.87530726, 1.95160199,
                1.97024652, 1.93070971, 1.74297716, 1.45822305, 1.10484158,
                0.82400562, 0.65350903, 0.56454676, 0.52291173, 0.50769533,
                0.50325891, 0.50109396, 0.50049024, 0.5001761 , 0.5       ]

CDR3B_ENTROPY = [0.50045576, 0.73260144, 0.97235865, 1.13315694, 2.        ,
                1.95656347, 1.90234182, 1.8979334 , 1.96717838, 1.98911718,
                1.98011194, 1.95488647, 1.82438546, 1.60232134, 1.30710213,
                1.00616478, 0.7760853 , 0.63962852, 0.56584238, 0.52940948,
                0.51247742, 0.50499323, 0.50186515, 0.50048074, 0.5       ]


class PeptideBatchConverter(object):

    def __init__(self, max_peptide_len=25):
        self.tokenizer = PeptideTokenizer()
        self.max_peptide_len = max_peptide_len

    def __call__(self, raw_batch: Sequence[str]):
        batch_size = len(raw_batch)
        seq_str_list = raw_batch

        seqs, tokens = peptide_batch_encoding(seq_str_list, self.tokenizer, batch_size, self.max_peptide_len)
        
        return {
            'seq': seqs,
            'token': tokens,
        }


class TCRBatchConverter(object):
    
    def __init__(self, max_cdr3_len=25, cdr12_encoding=True, drop_chain_prob=0.4):
        self.tokenizer = PairCDR3Tokenizer()
        self.max_cdr3_len = max_cdr3_len
        self.cdr12_encoding = cdr12_encoding
        self.drop_chain_prob = drop_chain_prob
           
    def drop_tcr_chain(self, batch, drop_prob):
        indices = random.sample(range(len(batch)), int(drop_prob * len(batch)))
        for idx in indices:
            if random.random() < 0.5:
                batch[idx]['cdr1a'] = np.nan
                batch[idx]['cdr2a'] = np.nan
                batch[idx]['cdr3a'] = np.nan
            else:
                batch[idx]['cdr1b'] = np.nan
                batch[idx]['cdr2b'] = np.nan
                batch[idx]['cdr3b'] = np.nan
        return batch
    
    def apply_entropy_weights(self, cdr3a_entropy, cdr3b_entropy, chain_token_mask):
        batch_size, seq_len = chain_token_mask.size()
        updated_weights = torch.ones((batch_size, seq_len), dtype=torch.float)
        
        alpha_mask = (chain_token_mask == 1)
        beta_mask = (chain_token_mask == 2)
        
        for i in range(batch_size):
            alpha_positions = alpha_mask[i].nonzero(as_tuple=True)[0]
            if len(alpha_positions) > 0:
                # Map each token position to the corresponding CDR3α entropy
                for idx, pos in enumerate(alpha_positions):
                    if idx < len(cdr3a_entropy):
                        updated_weights[i, pos] = cdr3a_entropy[idx]
            
            beta_positions = beta_mask[i].nonzero(as_tuple=True)[0]
            if len(beta_positions) > 0:
                # Map each token position to the corresponding CDR3β entropy
                for idx, pos in enumerate(beta_positions):
                    if idx < len(cdr3b_entropy):
                        updated_weights[i, pos] = cdr3b_entropy[idx]
                        
        return updated_weights
    
    def __call__(self, raw_batch: Sequence[str]):
        
        batch_ = self.drop_tcr_chain(raw_batch, self.drop_chain_prob)
        cdr3_tokens, chain_token_mask = paired_cdr3_batch_encoding(batch_, self.tokenizer, self.max_cdr3_len)
        
        batch = {
            'cdr3_token': cdr3_tokens,
            'chain_token_mask': chain_token_mask,
        }
        
        if self.cdr12_encoding:
            cdr12_feat_alpha, cdr12_feat_beta = paired_cdr12_batch_encoding(batch_)
            batch['cdr12_alpha_feat'] = cdr12_feat_alpha
            batch['cdr12_beta_feat'] = cdr12_feat_beta
            
        batch['entropy_weights'] = self.apply_entropy_weights(CDR3A_ENTROPY, CDR3B_ENTROPY, chain_token_mask)
        
        return batch
    

class PeptideMHCBacthConverter(object):
    def __init__(self, max_peptide_len=25, max_mhc_len=175):
        self.tokenizer = PeptideTokenizer()
        self.max_peptide_len = max_peptide_len
        self.max_mhc_len = max_mhc_len
        
        self.mh1_pseudo_pos = MH1_PSEUDO_POS
        self.mh2_pseudo_pos = MH2_PSEUDO_POS
        
    def __call__(self, raw_batch):
        batch_size = len(raw_batch)

        keys = raw_batch[0].keys()
        mhc_embs = np.stack([mhc_encoding(d['mhca_seq'], d['mhcb_seq'], d['class'], self.max_mhc_len) for d in raw_batch], axis=0)
        mhc_embs = torch.tensor(mhc_embs, dtype=torch.float)
        
        data = {key: [d[key] for d in raw_batch] for key in keys}
        _, peptide_tokens = peptide_batch_encoding(data['peptide'], self.tokenizer, batch_size, self.max_peptide_len)
        
        data['peptide_token'] = peptide_tokens
        
        mhc_pseudo_mask = torch.zeros((batch_size, self.max_mhc_len), dtype=torch.long)
        for i, mhc_class in enumerate(data['class']):
            if mhc_class == 'I':
                mhc_pseudo_mask[i, self.mh1_pseudo_pos] = 1
            elif mhc_class == 'II':
                mhc_pseudo_mask[i, self.mh2_pseudo_pos] = 1  
    
        data['mhc_embedding'] = mhc_embs
        data['mhc_pseudo_mask'] = mhc_pseudo_mask
        data['target'] = torch.tensor(data['target'], dtype=torch.float)
        
        return data
    
    
class TCRpMHCBatchConverter(object):
    def __init__(
        self,
        max_cdr3_len=25,
        max_peptide_len=25,
        max_mhc_len=175,
        cdr12_encoding=True,
        use_pmhc_struc_feat=False,
        pmhc_struc_feat_path=None
    ):
        self.tcr_tokenizer = PairCDR3Tokenizer()
        self.pep_tokenizer = PeptideTokenizer()
        
        self.max_cdr3_len = max_cdr3_len
        self.max_peptide_len = max_peptide_len
        self.max_mhc_len = max_mhc_len
        self.cdr12_encoding = cdr12_encoding
        self.use_pmhc_struc_feat = use_pmhc_struc_feat

        # use pmhc structural features
        if self.use_pmhc_struc_feat and pmhc_struc_feat_path is not None:
            with open(pmhc_struc_feat_path, "rb") as f:
                self.pmhc_struc_feat_dict = pickle.load(f)
        
        self.mh1_pseudo_pos = MH1_PSEUDO_POS
        self.mh2_pseudo_pos = MH2_PSEUDO_POS
        
    def __call__(self, raw_batch):
        batch_size = len(raw_batch)
        
        cdr3_tokens, chain_token_mask = paired_cdr3_batch_encoding(raw_batch, self.tcr_tokenizer, self.max_cdr3_len)
        batch = {
            'cdr3_token': cdr3_tokens,
            'chain_token_mask': chain_token_mask,
        }
        
        if self.cdr12_encoding:
            cdr12_feat_alpha, cdr12_feat_beta = paired_cdr12_batch_encoding(raw_batch)
            batch['cdr12_alpha_feat'] = cdr12_feat_alpha
            batch['cdr12_beta_feat'] = cdr12_feat_beta
        
        # batch['entropy_weights'] = self.apply_entropy_weights(CDR3A_ENTROPY, CDR3B_ENTROPY, chain_token_mask)
    
        _, peptide_tokens = peptide_batch_encoding([d['peptide'] for d in raw_batch], self.pep_tokenizer, batch_size, self.max_peptide_len)
        batch['peptide_token'] = peptide_tokens
        
        mhc_embs = torch.tensor(np.stack([mhc_encoding(d['mhca_seq'], d['mhcb_seq'], d['class'], self.max_mhc_len) for d in raw_batch], axis=0), dtype=torch.float)
        mhc_pseudo_mask = torch.zeros((batch_size, self.max_mhc_len), dtype=torch.long)
        for i, mhc_class in enumerate([d['class'] for d in raw_batch]):
            if mhc_class == 'I':
                mhc_pseudo_mask[i, self.mh1_pseudo_pos] = 1
            elif mhc_class == 'II':
                mhc_pseudo_mask[i, self.mh2_pseudo_pos] = 1  
    
        if self.use_pmhc_struc_feat:
            batch['pmhc_struc_feat'] = pmhc_struc_feat_batch_encoding(raw_batch, dim=peptide_tokens.shape[1] + 34, feat_dict=self.pmhc_struc_feat_dict)
            
        batch['mhc_embedding'] = mhc_embs
        batch['mhc_pseudo_mask'] = mhc_pseudo_mask
        
        if 'target' in raw_batch[0].keys():
            batch['target'] = torch.tensor([d['target'] for d in raw_batch], dtype=torch.float)
            weights = torch.ones_like(batch['target'])
            weights[[isinstance(d['cdr3a'], float) | isinstance(d['cdr3b'], float) for d in raw_batch]] = 0.5
            batch['weight'] = weights
            
        return batch
    

class TCRpMHCSamplingBatchConverter(object):
    def __init__(
        self,
        human_bg_tcr_path,
        mouse_bg_tcr_path,
        binding_matrix_path,
        pmhc_pool_path,
        tcr_pool_path,
        max_cdr3_len=25,
        max_peptide_len=25,
        max_mhc_len=175,
        cdr12_encoding=True,
        use_pmhc_struc_feat=False,
        pmhc_struc_feat_path=None,
        sample_bg_neg_ratio=2,
        sample_shuffle_neg_ratio=3
    ):
        self.tcr_tokenizer = PairCDR3Tokenizer()
        self.pep_tokenizer = PeptideTokenizer()
        
        self.max_cdr3_len = max_cdr3_len
        self.max_peptide_len = max_peptide_len
        self.max_mhc_len = max_mhc_len
        self.cdr12_encoding = cdr12_encoding
        
        self.mh1_pseudo_pos = MH1_PSEUDO_POS
        self.mh2_pseudo_pos = MH2_PSEUDO_POS
        
        self.sample_bg_neg_ratio = sample_bg_neg_ratio
        assert isinstance(self.sample_bg_neg_ratio, int) and self.sample_bg_neg_ratio > 0, "sample_bg_neg_ratio must be greater than 0"
        self.sample_shuffle_neg_ratio = sample_shuffle_neg_ratio
        assert isinstance(self.sample_shuffle_neg_ratio, int) and self.sample_shuffle_neg_ratio > 0, "sample_shuffle_neg_ratio must be greater than 0"
        
        self.human_bg_tcrs = pd.read_csv(human_bg_tcr_path)[['TRAV', 'CDR1A', 'CDR2A', 'CDR3A', 'TRBV', 'CDR1B', 'CDR2B', 'CDR3B']].drop_duplicates()
        self.mouse_bg_tcrs = pd.read_csv(mouse_bg_tcr_path)[['TRAV', 'CDR1A', 'CDR2A', 'CDR3A', 'TRBV', 'CDR1B', 'CDR2B', 'CDR3B']].drop_duplicates()
        
        self.pmhc_pool = pd.read_csv(pmhc_pool_path)
        self.tcr_pool = pd.read_csv(tcr_pool_path)
        self.tcr_pool.columns = [col.lower() for col in self.tcr_pool.columns]
        self.possible_binding_matrix = np.load(binding_matrix_path)
        
        self.tcr_columns = ['trav', 'cdr1a', 'cdr2a', 'cdr3a', 'trbv', 'cdr1b', 'cdr2b', 'cdr3b']
        
        self.use_pmhc_struc_feat = use_pmhc_struc_feat
        if self.use_pmhc_struc_feat and pmhc_struc_feat_path is not None:
            with open(pmhc_struc_feat_path, "rb") as f:
                self.pmhc_struc_feat_dict = pickle.load(f)
    
    def sample_bg_neg_pairs(self, batch: dict):
        
        batch = pd.DataFrame(batch)
        
        batch_human = batch[batch['organism'] == 'human']
        batch_mouse = batch[batch['organism'] == 'mouse']
        
        new_batch = []
        for sub_batch in [batch_human, batch_mouse]:
            if len(sub_batch) == 0:
                continue
            else:
                sample_size = len(sub_batch) * self.sample_bg_neg_ratio
                bg_tcrs = self.human_bg_tcrs if sub_batch['organism'].iloc[0] == 'human' else self.mouse_bg_tcrs
                sampled_bg_tcrs = bg_tcrs.sample(n=sample_size, replace=False).reset_index(drop=True)
                
                if sample_size == 0:
                    return sampled_bg_tcrs.to_dict('list')
                        
                sub_batch_repeated = pd.concat([sub_batch] * self.sample_bg_neg_ratio, ignore_index=True)
                
                # Vectorized approach - replace all relevant columns at once
                tcr_cols_lower = self.tcr_columns
                tcr_cols_upper = [col.upper() for col in tcr_cols_lower]

                # Create masks for notna values for each column
                masks = {col: sub_batch_repeated[col].notna() for col in tcr_cols_lower}

                # Apply the replacements in a single vectorized operation
                for lcol, ucol in zip(tcr_cols_lower, tcr_cols_upper):
                    mask = masks[lcol]
                    if mask.any():  # Only perform assignment if there are values to replace
                        sub_batch_repeated.loc[mask, lcol] = sampled_bg_tcrs.loc[mask.index[mask], ucol].values
                    
                new_batch.append(sub_batch_repeated)
        
        return pd.concat(new_batch, ignore_index=True).to_dict('list')
    
    
    def sample_shuffle_neg_pairs(self, batch: dict):
        
        batch = pd.DataFrame(batch)
        
        new_batch = []
        for i in range(len(batch)):
            peptide = batch['peptide'].iloc[i]
            mhc = batch['mhc'].iloc[i]
            mhca = batch['mhca'].iloc[i]
            mhcb = batch['mhcb'].iloc[i]
            organism = batch['organism'].iloc[i]
            
            # extract pmhc index and corresponding non-binding tcrs
            pmhc_idx = self.pmhc_pool[(self.pmhc_pool['Peptide'] == peptide) & (self.pmhc_pool['MHCA'] == mhca) & (self.pmhc_pool['MHCB'] == mhcb)].index[0]
            # restric the organism
            non_binding_tcrs = self.tcr_pool[(self.possible_binding_matrix[:, pmhc_idx] == 0) & (self.tcr_pool['organism'] == organism)]
            
            # choose non binding subpopulation based on whether cdr3a and cdr3b is nan.
            if not isinstance(batch['cdr3a'].iloc[i], str):
                non_binding_tcrs = non_binding_tcrs[non_binding_tcrs['cdr3a'].isna()]
            elif not isinstance(batch['cdr3b'].iloc[i], str):
                non_binding_tcrs = non_binding_tcrs[non_binding_tcrs['cdr3b'].isna()]
            else:
                non_binding_tcrs = non_binding_tcrs[non_binding_tcrs.notna().all(axis=1)]
            
            # sample non-binding tcrs
            # make sure number of sampled tcrs is greater than or equal to sample_shuffle_neg_ratio
            if len(non_binding_tcrs) < self.sample_shuffle_neg_ratio:
                sampled_non_binding_tcrs = non_binding_tcrs.sample(len(non_binding_tcrs), replace=False).reset_index(drop=True)
            else:
                sampled_non_binding_tcrs = non_binding_tcrs.sample(self.sample_shuffle_neg_ratio, replace=False).reset_index(drop=True)

            sampled_non_binding_tcrs['peptide'] = peptide
            sampled_non_binding_tcrs['mhc'] = mhc
            sampled_non_binding_tcrs['mhca'] = mhca
            sampled_non_binding_tcrs['mhcb'] = mhcb
            sampled_non_binding_tcrs['class'] = 'I' if mhcb == 'b2m' else 'II'
            sampled_non_binding_tcrs['mhca_seq'] = batch['mhca_seq'].iloc[i]
            sampled_non_binding_tcrs['mhcb_seq'] = batch['mhcb_seq'].iloc[i]
            
            new_batch.append(sampled_non_binding_tcrs)
        
        return pd.concat(new_batch, ignore_index=True).to_dict('list')
                
    def __call__(self, raw_batch):
        
        batch_size = len(raw_batch)
        
        # transform raw batch into a dictionary of lists
        keys = raw_batch[0].keys()        
        data = {key: [d[key] for d in raw_batch] for key in keys}
        
        bg_negative_data = self.sample_bg_neg_pairs(data)
        shuffle_negative_data = self.sample_shuffle_neg_pairs(data)
        
        # merge data and bg_negative data add 'binding=1' or 'binding=0' (TEST only use shuffled negatives)
        for key in data.keys():
            data[key] = data[key] + bg_negative_data[key] + shuffle_negative_data[key]
        
        data['target'] = [1] * batch_size + [0] * len(bg_negative_data['peptide'] + shuffle_negative_data['peptide'])
        
        data_list = [dict(zip(keys, v)) for v in zip(*data.values())]
        cdr3_tokens, chain_token_mask = paired_cdr3_batch_encoding(data_list, self.tcr_tokenizer, self.max_cdr3_len)
        batch = {
            'cdr3_token': cdr3_tokens,
            'chain_token_mask': chain_token_mask,
        }
        
        if self.cdr12_encoding:
            cdr12_feat_alpha, cdr12_feat_beta = paired_cdr12_batch_encoding(data_list)
            batch['cdr12_alpha_feat'] = cdr12_feat_alpha
            batch['cdr12_beta_feat'] = cdr12_feat_beta
    
        _, peptide_tokens = peptide_batch_encoding(data['peptide'], self.pep_tokenizer, len(data_list), self.max_peptide_len)
        batch['peptide_token'] = peptide_tokens
        
        mhc_embs = torch.tensor(np.stack([mhc_encoding(d['mhca_seq'], d['mhcb_seq'], d['class'], self.max_mhc_len) for d in data_list], axis=0), dtype=torch.float)
        mhc_pseudo_mask = torch.zeros((len(data_list), self.max_mhc_len), dtype=torch.long)
        
        for i, mhc_class in enumerate(data['class']):
            if mhc_class == 'I':
                mhc_pseudo_mask[i, self.mh1_pseudo_pos] = 1
            elif mhc_class == 'II':
                mhc_pseudo_mask[i, self.mh2_pseudo_pos] = 1  
    
        batch['mhc_embedding'] = mhc_embs
        batch['mhc_pseudo_mask'] = mhc_pseudo_mask
        
        # TODO: featurize pmhc as graph (https://github.com/drorlab/gvp-pytorch)
        if self.use_pmhc_struc_feat:
            batch['pmhc_struc_feat'] = pmhc_struc_feat_batch_encoding(data_list, dim=peptide_tokens.shape[1] + 34, feat_dict=self.pmhc_struc_feat_dict) # peptide len + mhc pseudo seq len (34)
        
        batch['target'] = torch.tensor(data['target'], dtype=torch.float)

        # Optional: set weight for loss function (single chain cdr3a/b is nan)
        weights = torch.ones_like(batch['target'])
        weights[[isinstance(d['cdr3a'], float) | isinstance(d['cdr3b'], float) for d in data_list]] = 0.5
        batch['weight'] = weights
            
        return batch
