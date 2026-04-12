from torch.utils.data import Dataset
import pandas as pd
import numpy as np
import json

from ..utils.encoding import mhc_allele_to_seq


class PeptideDataset(Dataset):

    def __init__(self, data_path):
        super().__init__()
        self.data_path = data_path
        self.peptides = self.load_data(self.data_path)

    def load_data(self, path):
        return [line.strip() for line in open(path, 'r')]

    def __getitem__(self, idx):
        seq = self.peptides[idx]
        return seq

    def __len__(self):
        return len(self.peptides)
    
    
class TCRDataset(Dataset):
    
    def __init__(self, data_path, subsampling=False, subsample_size=1000):
        self.data_path = data_path
        self.subsampling = subsampling
        self.subsample_size = subsample_size
        self.data = self.load_data(self.data_path)
        
    def load_data(self, path):
        data = pd.read_csv(path)
        if self.subsampling:
            data = self.subsample(data, self.subsample_size)

        return {
            'cdr1a': data['cdr1a'],
            'cdr1b': data['cdr1b'],
            'cdr2a': data['cdr2a'],
            'cdr2b': data['cdr2b'],
            'cdr3a': data['cdr3a'],
            'cdr3b': data['cdr3b'],
        }
    
    # Subsample data to have maximum number of samples (1,000) for each CDR1-CDR2 pair
    def subsample(self, data, subsample_size):
        grouped = data.groupby(['cdr1a', 'cdr1b', 'cdr2a', 'cdr2b'])
        subsampled_data = []
        for group in grouped:
            if len(group[1]) > subsample_size:
                subsampled_data.append(group[1].sample(subsample_size))
            else:
                subsampled_data.append(group[1])
        return pd.concat(subsampled_data, ignore_index=True)
        
    def __getitem__(self, idx):
        return {key:value[idx] for key, value in self.data.items()}

    def __len__(self):
        return len(self.data['cdr3b'])


class PeptideMHCDataset(Dataset):
    
    def __init__(self, data_path, mhc_lib_path):
        super().__init__()
        self.data_path = data_path
        with open(mhc_lib_path, 'r') as f:
            self.mhc_library = json.load(f)
        self.data = self.load_data()
        
    def load_data(self):
        rawdata = pd.read_csv(self.data_path)
        epitope = rawdata['epitope']
        mhc_class = rawdata['class']
        mhc_alpha = rawdata['mhca']
        mhc_beta = rawdata['mhcb']
        target = rawdata['binding']
        
        return {
            'peptide': epitope,
            'class': mhc_class,
            'mhca': mhc_alpha,
            'mhcb': mhc_beta,
            'target': target
        }
    
    def transform_mhc_allele(self, mhc_allele):
        if mhc_allele == 'b2m':
            return None
        else:
            mhc_seq = mhc_allele_to_seq(mhc_allele.replace('HLA-', ''), self.mhc_library)
            return mhc_seq
        
    def __getitem__(self, idx):
        data = {key:value[idx] for key, value in self.data.items()}
        
        data['mhca_seq'] = self.transform_mhc_allele(data['mhca'])
        data['mhcb_seq'] = self.transform_mhc_allele(data['mhcb'])
        return data
        
    def __len__(self):
        return len(self.data['target'])
    
    
class TCRpMHCDataset(Dataset):
    
    def __init__(self, data_path, mhc_lib_path, v_gene_lib_path, use_cdr12_columns=False, include_target=False):
        super().__init__()
        
        self.include_target = include_target
        self.use_cdr12_columns = use_cdr12_columns
        
        self.data = self.load_data(data_path)
        
        with open(mhc_lib_path, 'r') as f:
            self.mhc_library = json.load(f)
        
        self.v_gene_library = {}
        v_gene_library = pd.read_csv(v_gene_lib_path)
        
        for org in ['human', 'mouse']:
            self.v_gene_library[org] = v_gene_library[v_gene_library['organism'] == org]
    
    def load_data(self, data_path):
        if isinstance(data_path, pd.DataFrame):
            rawdata = data_path
        else:
            rawdata = pd.read_csv(data_path)

        data = {
            'peptide': rawdata['Peptide'],
            'mhc': rawdata['MHC'],
            'mhca': rawdata['MHCA'],
            'mhcb': rawdata['MHCB'],
            'organism': rawdata['Organism'],
            'trav': rawdata['TRAV'],
            'cdr3a': rawdata['CDR3A'],
            'trbv': rawdata['TRBV'],
            'cdr3b': rawdata['CDR3B']
        }
        
        if self.include_target:
            data['target'] = rawdata['Binding']

        if self.use_cdr12_columns:
            data['cdr1a'] = rawdata['CDR1A']
            data['cdr2a'] = rawdata['CDR2A']
            data['cdr1b'] = rawdata['CDR1B']
            data['cdr2b'] = rawdata['CDR2B']
            
        return data
    
    def transform_mhc_allele(self, mhc_allele):
        if mhc_allele == 'b2m':
            return None
        else:
            mhc_seq = mhc_allele_to_seq(mhc_allele.replace('HLA-', ''), self.mhc_library)
            return mhc_seq
    
    def extract_cdr12_seq(self, v_gene, organism):
        if isinstance(v_gene, str):
            if len(v_gene.split('*')) == 1:
                v_gene = v_gene + '*01'
                
            gene_df = self.v_gene_library[organism]
            assert v_gene in gene_df['id'].values
            
            cdrs = gene_df[gene_df['id'] == v_gene]['cdrs'].values[0]
            cdr1, cdr2 = cdrs.split(';')[0], cdrs.split(';')[1]
            return cdr1.replace('.', ''), cdr2.replace('.', '')
        else:
            return np.nan, np.nan
    
    def __getitem__(self, idx):
        data = {key:value[idx] for key, value in self.data.items()}
        
        data['mhca_seq'] = self.transform_mhc_allele(data['mhca'])
        data['mhcb_seq'] = self.transform_mhc_allele(data['mhcb'])
        data['class'] = 'I' if data['mhcb'] == 'b2m' else 'II'
        
        # cdr12 sequence extraction
        if not self.use_cdr12_columns:
            data['cdr1a'], data['cdr2a'] = self.extract_cdr12_seq(data['trav'], data['organism'])
            data['cdr1b'], data['cdr2b'] = self.extract_cdr12_seq(data['trbv'], data['organism'])
        
        return data
        
    def __len__(self):
        return len(self.data['peptide'])
        