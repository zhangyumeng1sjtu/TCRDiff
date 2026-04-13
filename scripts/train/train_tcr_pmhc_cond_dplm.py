from argparse import ArgumentParser
import os

import torch
from torch.utils.data import DataLoader, WeightedRandomSampler, RandomSampler
from torchinfo import summary
import pandas as pd
import numpy as np

import sys
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)
    
from TCRDiff.utils import set_seed, load_config
from TCRDiff.dataset import TCRpMHCDataset, TCRpMHCBatchConverter
from TCRDiff.trainer import ConditionalTCRDPLMTrainer


def main(args):
    # load config
    config_path = args.config
    config = load_config(config_path)

    if not os.path.exists(config.training.log_dir):
        os.makedirs(config.training.log_dir)
    os.system(f'cp {args.config} {config.training.log_dir}/config.yml')

    set_seed(config.training.seed)

    train_data = TCRpMHCDataset(config.data.train_data_path, config.data.mhc_lib_path, config.data.v_gene_lib_path)
    val_data = TCRpMHCDataset(config.data.val_data_path, config.data.mhc_lib_path, config.data.v_gene_lib_path)
    test_data = TCRpMHCDataset(config.data.test_data_path, config.data.mhc_lib_path, config.data.v_gene_lib_path, use_cdr12_columns=True)
    
    batch_converter = TCRpMHCBatchConverter(max_cdr3_len=config.data.max_cdr3_len,
                                            max_peptide_len=config.data.max_peptide_len,
                                            max_mhc_len=config.data.max_mhc_len)
    Trainer = ConditionalTCRDPLMTrainer(config)
    

    print(f'Train data size: {len(train_data)}')
    print(f'Val data size: {len(val_data)}')
    print(f'Test data size: {len(test_data)}')
    
    train_loader = DataLoader(train_data, batch_size=config.training.batch_size, collate_fn=batch_converter,
                            num_workers=config.training.num_workers, shuffle=True)
    val_loader = DataLoader(val_data, batch_size=config.training.batch_size, collate_fn=batch_converter,
                            num_workers=config.training.num_workers, shuffle=False)
    test_loader = DataLoader(test_data, batch_size=config.training.batch_size, collate_fn=batch_converter,
                             num_workers=config.training.num_workers, shuffle=False)
    
    batch = next(iter(val_loader))
    for key, value in batch.items():
        if isinstance(value, torch.Tensor):
            batch[key] = value.to(Trainer.device)
            
    summary(Trainer.model, input_data = [batch])

    Trainer.fit(train_loader, val_loader)
    Trainer.test(test_loader, model_location=os.path.join(config.training.log_dir, 'checkpoint.pt'))
    
    # Test prediction results
    val_pred = Trainer.predict(val_loader, model_location=os.path.join(config.training.log_dir, 'checkpoint.pt'), noise = 'full_mask')
    test_pred = Trainer.predict(test_loader, model_location=os.path.join(config.training.log_dir, 'checkpoint.pt'), noise = 'full_mask')
    
    print(val_pred['accuracy'], val_pred['median_accuracy'])
    print(test_pred['accuracy'], test_pred['median_accuracy'])

if __name__ == '__main__':
    parser = ArgumentParser()

    parser.add_argument('--config', type=str, default='configs/config-train-tcr-pmhc-dplm.yml')
    args = parser.parse_args()

    main(args)
