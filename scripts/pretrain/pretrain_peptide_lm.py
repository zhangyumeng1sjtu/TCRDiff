from argparse import ArgumentParser
import os

import torch
from torch.utils.data import DataLoader
from torchinfo import summary

import sys
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from TCRDiff.utils import set_seed, load_config
from TCRDiff.dataset import PeptideDataset, PeptideBatchConverter
from TCRDiff.trainer import PeptideMLMTrainer


def main(args):
    # load config
    config_path = args.config
    config = load_config(config_path)

    if not os.path.exists(config.training.log_dir):
        os.makedirs(config.training.log_dir)
    os.system(f'cp {args.config} {config.training.log_dir}/config.yml')

    set_seed(config.training.seed)
    
    train_data = PeptideDataset(config.data.train_data_path)
    val_data = PeptideDataset(config.data.val_data_path)
    test_data = PeptideDataset(config.data.test_data_path)
    
    batch_converter = PeptideBatchConverter(max_peptide_len=config.data.max_peptide_len)
    Trainer = PeptideMLMTrainer(config)

    print(f'Train data size: {len(train_data)}')
    print(f'Val data size: {len(val_data)}')
    print(f'Test data size: {len(test_data)}')

    train_loader = DataLoader(train_data, batch_size=config.training.batch_size, collate_fn=batch_converter,
                              num_workers=config.training.num_workers, shuffle=True)
    val_loader = DataLoader(val_data, batch_size=config.training.batch_size, collate_fn=batch_converter,
                            num_workers=config.training.num_workers, shuffle=False)
    test_loader = DataLoader(test_data, batch_size=config.training.batch_size, collate_fn=batch_converter,
                             num_workers=config.training.num_workers, shuffle=False)

    summary(Trainer.model, [(1,26)], dtypes=[torch.int])
    
    Trainer.fit(train_loader, val_loader)
    Trainer.test(test_loader, model_location=os.path.join(config.training.log_dir, 'checkpoint.pt'))


if __name__ == '__main__':
    parser = ArgumentParser()

    parser.add_argument('--config', type=str, default='configs/config-pretrain-peptide-lm.yml')
    args = parser.parse_args()

    main(args)
