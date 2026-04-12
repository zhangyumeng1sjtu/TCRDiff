import argparse

import torch
import pandas as pd
import numpy as np
from tqdm import tqdm
from torch.utils.data import DataLoader
from sklearn.metrics import roc_auc_score, average_precision_score

from TCRDiff.dataset import TCRpMHCDataset, TCRpMHCBatchConverter
from TCRDiff.utils import set_seed, load_config
from TCRDiff.model import TCRpMHCPairFormer, TCRpMHCCoembeddingPairFormer, TCRpMHCCoembeddingModel


# params
parser = argparse.ArgumentParser()
parser.add_argument('--dirname', type=str, required=True)
parser.add_argument('--model_name', type=str, required=True)
parser.add_argument('--mhc_class', type=str, required=True) # [i, ii]
parser.add_argument('--gpu_device', type=int, default=0)
args = parser.parse_args()

# load config
config_path = f"{args.dirname}/config.yml"
model_location = f"{args.dirname}/checkpoint.pt"

config = load_config(config_path)
device = torch.device(f"cuda:{args.gpu_device}" if torch.cuda.is_available() else "cpu")
set_seed(config.training.seed)

# load data
val_data_path = f"data/tcrpmhc/class_{args.mhc_class}/validation_tcr_pmhc_data_class_{args.mhc_class}.csv"
val_data = TCRpMHCDataset(data_path=val_data_path,
                           mhc_lib_path="data/pmhc/mhc_lib.json",
                           v_gene_lib_path="data/tcrpmhc/v_gene_lib.csv",
                           use_cdr12_columns=True,
                           include_target=True
)
batch_converter = TCRpMHCBatchConverter(max_cdr3_len=config.data.max_cdr3_len,
                                        max_peptide_len=config.data.max_peptide_len,
                                        max_mhc_len=config.data.max_mhc_len,
                                        use_pmhc_struc_feat=True
)
val_loader = DataLoader(val_data, batch_size=config.training.test_batch_size, collate_fn=batch_converter, num_workers=config.training.num_workers, shuffle=False)

# load model
if model_name == 'pairformer':
    model = TCRpMHCPairFormer(config.model.tcr_model, config.model.pmhc_model, **config.model.pairformer)
elif model_name == 'pairformer_cl':
    model = TCRpMHCCoembeddingPairFormer(config.model.tcr_model, config.model.pmhc_model, **config.model.pairformer)
elif model_name == 'coembedding':
    model = TCRpMHCCoembeddingModel(config.model.tcr_model, config.model.pmhc_model, **config.model.coembedding)
    
model.load_state_dict(torch.load(model_location, weights_only=True, map_location='cpu'))
model.to(device)


# Val    
pred = []
model.eval()
with torch.no_grad():
    for batch in tqdm(val_loader):
        for key, value in batch.items():
            if isinstance(value, torch.Tensor):
                batch[key] = value.to(device)
                
        out = model(batch)
        if model_name == 'pairformer':
            pred.append(out.squeeze(-1).detach().cpu().numpy())
        elif model_name == 'pairformer_cl':
            pred.append(out['logits'].squeeze(-1).detach().cpu().numpy())
        elif model_name == 'coembedding':
            pred.append(out['logits'].squeeze(-1).detach().cpu().numpy())
        
pred = np.concatenate(pred)

val_df = pd.read_csv(val_data_path)
val_df['Pred'] = pred

# out_path = f"{dirname}/val_pred.csv"
out_path = f"{dirname}/val_pred_{mhc_class}.csv"
val_df.to_csv(out_path, index=False)

auc01 = val_df.groupby(['Peptide', 'MHC']).apply(lambda row: roc_auc_score(row['Binding'], row['Pred'], max_fpr=0.1))
auc = val_df.groupby(['Peptide', 'MHC']).apply(lambda row: roc_auc_score(row['Binding'], row['Pred']))
aupr = val_df.groupby(['Peptide', 'MHC']).apply(lambda row: average_precision_score(row['Binding'], row['Pred']))

print(f"Val AUC: {roc_auc_score(val_df['Binding'], val_df['Pred']):.3f}")
print(f"Val AUC@0.1: {roc_auc_score(val_df['Binding'], val_df['Pred'], max_fpr=0.1):.3f}")
print(f"Val AUPR: {average_precision_score(val_df['Binding'], val_df['Pred']):.3f}")

print(f"Val AVG AUC: {auc.mean():.3f}")
print(f"Val AVG AUC@0.1: {auc01.mean():.3f}")
print(f"Val AVG AUPR: {aupr.mean():.3f}")


# Test
if mhc_class == 'i':
    test_data_path = f"data/tcrpmhc/class_{args.mhc_class}/immrep23_test_data.csv"
    test_data = TCRpMHCDataset(data_path=test_data_path,
                            mhc_lib_path="data/pmhc/mhc_lib.json",
                            v_gene_lib_path="data/tcrpmhc/v_gene_lib.csv",
                            use_cdr12_columns=True,
                            include_target=True
    )
    test_loader = DataLoader(test_data, batch_size=config.training.test_batch_size, collate_fn=batch_converter, num_workers=config.training.num_workers, shuffle=False)

    pred = []
    model.eval()
    with torch.no_grad():
        for batch in tqdm(test_loader):
            for key, value in batch.items():
                if isinstance(value, torch.Tensor):
                    batch[key] = value.to(device)
                    
            out = model(batch)
            if model_name == 'pairformer':
                pred.append(out.squeeze(-1).detach().cpu().numpy())
            elif model_name == 'pairformer_cl':
                pred.append(out['logits'].squeeze(-1).detach().cpu().numpy())
            elif model_name == 'coembedding':
                pred.append(out['logits'].squeeze(-1).detach().cpu().numpy())
            
    pred = np.concatenate(pred)

    test_df = pd.read_csv(test_data_path)
    test_df['Pred'] = pred

    test_df = test_df[test_df['Usage'] == 'Private']

    out_path = f"{dirname}/test_pred.csv"
    test_df.to_csv(out_path, index=False)

    auc01 = test_df.groupby(['Peptide', 'MHC']).apply(lambda row: roc_auc_score(row['Binding'], row['Pred'], max_fpr=0.1))
    auc = test_df.groupby(['Peptide', 'MHC']).apply(lambda row: roc_auc_score(row['Binding'], row['Pred']))
    aupr = test_df.groupby(['Peptide', 'MHC']).apply(lambda row: average_precision_score(row['Binding'], row['Pred']))

    print(f"Test AUC: {roc_auc_score(test_df['Binding'], test_df['Pred']):.3f}")
    print(f"Test AUC@0.1: {roc_auc_score(test_df['Binding'], test_df['Pred'], max_fpr=0.1):.3f}")
    print(f"Test AUPR: {average_precision_score(test_df['Binding'], test_df['Pred']):.3f}")

    print(f"Test AVG AUC: {auc.mean():.3f}")
    print(f"Test AVG AUC@0.1: {auc01.mean():.3f}")
    print(f"Test AVG AUPR: {aupr.mean():.3f}")
