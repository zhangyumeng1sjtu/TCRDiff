from argparse import ArgumentParser
import os
import torch
from torch.utils.data import DataLoader

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

    test_df = pd.read_csv(args.input_data_path)
    test_df.loc[:, 'Pred'] = pd.read_csv(args.saveto + '/predictions.csv')['Pred']
    test_df.to_csv(args.saveto + '/results.csv', index=False)


if __name__ == '__main__':
    parser = ArgumentParser()

    parser.add_argument('--config', type=str, default='configs/config-train-tcr-pmhc-model.yml')
    parser.add_argument('--model_location', type=str, default=None)
    parser.add_argument('--input_data_path', type=str, required=True)
    parser.add_argument('--saveto', type=str, required=True)
    parser.add_argument('--gpu_device', type=int, default=0)
    parser.add_argument('--use_cdr12_columns', action='store_true')
    
    args = parser.parse_args()

    main(args)
    

