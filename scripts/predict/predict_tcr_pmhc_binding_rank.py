from argparse import ArgumentParser
import os
import torch
from torch.utils.data import DataLoader
from scipy import stats
import pandas as pd

from TCRDiff.dataset import TCRpMHCDataset, TCRpMHCBatchConverter
from TCRDiff.utils import set_seed, load_config
from TCRDiff.trainer import TCRpMHCBindingTrainer, TCRpMHCCoembeddingTrainer


def main(args):
    # load config
    config_path = args.config
    config = load_config(config_path)
    config.training.gpu_device = args.gpu_device
    model_location = args.model_location if args.model_location is not None else os.path.join(config.training.log_dir, 'checkpoint.pt')
    
    set_seed(config.training.seed)
    test_data = TCRpMHCDataset(
        args.input_data_path,
        config.data.mhc_lib_path,
        config.data.v_gene_lib_path,
        use_cdr12_columns=args.use_cdr12_columns
    )
    batch_converter = TCRpMHCBatchConverter(
        max_cdr3_len=config.data.max_cdr3_len,
        max_peptide_len=config.data.max_peptide_len,
        max_mhc_len=config.data.max_mhc_len
    ) 
    
    Trainer = TCRpMHCBindingTrainer(config)
    # Trainer = TCRpMHCCoembeddingTrainer(config)

    print(f'Test data size: {len(test_data)}')
    test_loader = DataLoader(
        test_data,
        batch_size=config.training.test_batch_size,
        collate_fn=batch_converter,
        num_workers=config.training.num_workers,
        shuffle=False
    )

    os.makedirs(args.saveto, exist_ok=True)
    Trainer.predict(
        test_loader,
        model_location=model_location,
        out_dir=args.saveto
    )

    bg_tcr_df = pd.read_csv(args.background_data_path)
    sampled_bg_df = bg_tcr_df.sample(n=args.num_bg_samples, replace=False).reset_index(drop=True)

    # compute background predictions for each peptide-mhc combination
    test_df = pd.read_csv(args.input_data_path)
    unique_peptide_mhc = test_df[['Peptide', 'MHC', 'MHCA', 'MHCB', 'Organism']].drop_duplicates()
    ori_pred = pd.read_csv(args.saveto + '/predictions.csv')['Pred']
    test_df['Pred'] = ori_pred

    results = []
    for i in range(len(unique_peptide_mhc)):
        peptide = unique_peptide_mhc['Peptide'].iloc[i]
        mhc = unique_peptide_mhc['MHC'].iloc[i]
        mhca = unique_peptide_mhc['MHCA'].iloc[i]
        mhcb = unique_peptide_mhc['MHCB'].iloc[i]
        organism = unique_peptide_mhc['Organism'].iloc[i]

        sampled_bg_df['Peptide'] = peptide
        sampled_bg_df['MHC'] = mhc
        sampled_bg_df['MHCA'] = mhca
        sampled_bg_df['MHCB'] = mhcb
        sampled_bg_df['Organism'] = organism
        bg_test_data = TCRpMHCDataset(
            sampled_bg_df,
            config.data.mhc_lib_path,
            config.data.v_gene_lib_path,
            use_cdr12_columns=args.use_cdr12_columns
        )
        bg_test_loader = DataLoader(
            bg_test_data,
            batch_size=config.training.test_batch_size,
            collate_fn=batch_converter,
            num_workers=config.training.num_workers,
            shuffle=False
        )
        
        os.makedirs(args.saveto + f'/bg_{peptide}_{mhc}', exist_ok=True)
        Trainer.predict(
            bg_test_loader,
            model_location=model_location,
            out_dir=args.saveto + f'/bg_{peptide}_{mhc}'
        )

        test_sub_df = test_df[(test_df['Peptide'] == peptide) & (test_df['MHC'] == mhc)].reset_index(drop=True)
        bg_pred = pd.read_csv(args.saveto + f'/bg_{peptide}_{mhc}/predictions.csv')['Pred']
        test_sub_df.loc[:, 'Percentile'] = stats.percentileofscore(bg_pred, test_sub_df['Pred'])
  
        results.append(test_sub_df)

    results_df = pd.concat(results)
    results_df.to_csv(args.saveto + '/results.csv', index=False)


if __name__ == '__main__':
    parser = ArgumentParser()

    parser.add_argument('--config', type=str, default='configs/config-train-tcr-pmhc-model.yml')
    parser.add_argument('--model_location', type=str, default=None)
    parser.add_argument('--input_data_path', type=str, required=True)
    parser.add_argument('--background_data_path', type=str, required=True)
    parser.add_argument('--num_bg_samples', type=int, default=10000)
    parser.add_argument('--saveto', type=str, required=True)
    parser.add_argument('--gpu_device', type=int, default=0)
    parser.add_argument('--use_cdr12_columns', action='store_true')
    
    args = parser.parse_args()

    main(args)
    