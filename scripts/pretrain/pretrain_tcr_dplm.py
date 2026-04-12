from argparse import ArgumentParser
import os

import torch
from torch.utils.data import DataLoader
from torchinfo import summary

from TCRDiff.utils import set_seed, load_config
from TCRDiff.dataset import TCRDataset, TCRBatchConverter
from TCRDiff.trainer import TCRDPLMTrainer


def main(args):
    # load config
    config_path = args.config
    config = load_config(config_path)

    if not os.path.exists(config.training.log_dir):
        os.makedirs(config.training.log_dir)
    os.system(f'cp {args.config} {config.training.log_dir}/config.yml')

    set_seed(config.training.seed)
    
    # train_data = TCRDataset(config.data.train_data_path, subsampling=True, subsample_size=1000)
    train_data = TCRDataset(config.data.train_data_path)
    val_data = TCRDataset(config.data.val_data_path)
    test_data = TCRDataset(config.data.test_data_path)
    
    batch_converter = TCRBatchConverter(max_cdr3_len=config.data.max_cdr3_len, drop_chain_prob=config.data.drop_chain_prob)
    Trainer = TCRDPLMTrainer(config)

    print(f'Train data size: {len(train_data)}')
    print(f'Val data size: {len(val_data)}')
    print(f'Test data size: {len(test_data)}')

    train_loader = DataLoader(train_data, batch_size=config.training.batch_size, collate_fn=batch_converter,
                              num_workers=config.training.num_workers, shuffle=True)
    val_loader = DataLoader(val_data, batch_size=config.training.batch_size, collate_fn=batch_converter,
                            num_workers=config.training.num_workers, shuffle=False)
    test_loader = DataLoader(test_data, batch_size=config.training.batch_size, collate_fn=batch_converter,
                             num_workers=config.training.num_workers, shuffle=False)

    # summary(Trainer.model, [(1,52), (1,52), (1, 16, 5), (1, 13, 5)], dtypes=[torch.int, torch.int, torch.float, torch.float])
    
    Trainer.fit(train_loader, val_loader)
    Trainer.test(test_loader, model_location=os.path.join(config.training.log_dir, 'checkpoint.pt'))
    
    # Test prediction results
    # val_pred = Trainer.predict(val_loader, model_location=os.path.join(config.training.log_dir, 'checkpoint.pt'), noise = 'full_mask')
    # test_pred = Trainer.predict(test_loader, model_location=os.path.join(config.training.log_dir, 'checkpoint.pt'), noise = 'full_mask')
    
    # print(val_pred['accuracy'], val_pred['median_accuracy'])
    # print(test_pred['accuracy'], test_pred['median_accuracy'])


if __name__ == '__main__':
    parser = ArgumentParser()

    parser.add_argument('--config', type=str, default='configs/config-pretrain-tcr-dplm.yml')
    args = parser.parse_args()

    main(args)
