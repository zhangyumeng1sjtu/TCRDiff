from argparse import ArgumentParser
import os

from torch.utils.data import DataLoader
from torchinfo import summary

from TCRDiff.utils import set_seed, load_config
from TCRDiff.dataset import PeptideMHCDataset, PeptideMHCBacthConverter
from TCRDiff.trainer import PeptideMHCTrainer, PeptideMHCELTrainer


def main(args):
    # load config
    config_path = args.config
    config = load_config(config_path)

    if not os.path.exists(config.training.log_dir):
        os.makedirs(config.training.log_dir)
    os.system(f'cp {args.config} {config.training.log_dir}/config.yml')

    set_seed(config.training.seed)
    
    train_data = PeptideMHCDataset(config.data.train_data_path, config.data.mhc_lib_path)
    val_data = PeptideMHCDataset(config.data.val_data_path, config.data.mhc_lib_path)
    test_data = PeptideMHCDataset(config.data.test_data_path, config.data.mhc_lib_path)
    
    batch_converter = PeptideMHCBacthConverter(max_peptide_len=config.data.max_peptide_len, max_mhc_len=config.data.max_mhc_len)
    Trainer = PeptideMHCTrainer(config)
    # Trainer = PeptideMHCELTrainer(config)

    print(f'Train data size: {len(train_data)}')
    print(f'Val data size: {len(val_data)}')
    print(f'Test data size: {len(test_data)}')
    
    train_loader = DataLoader(train_data, batch_size=config.training.batch_size, collate_fn=batch_converter,
                              num_workers=config.training.num_workers, shuffle=True)
    val_loader = DataLoader(val_data, batch_size=config.training.batch_size, collate_fn=batch_converter,
                            num_workers=config.training.num_workers, shuffle=False)
    test_loader = DataLoader(test_data, batch_size=config.training.batch_size, collate_fn=batch_converter,
                             num_workers=config.training.num_workers, shuffle=False)

    batch = next(iter(test_loader))
    pep_labels = batch['peptide_token'].to(Trainer.device)
    mhc_embeds = batch['mhc_embedding'].to(Trainer.device)
    mhc_pseudo_mask = batch['mhc_pseudo_mask'].to(Trainer.device)
    
    summary(Trainer.model, input_data = [pep_labels[0,:].unsqueeze(0),
                                         mhc_embeds[0,:,:].unsqueeze(0),
                                         mhc_pseudo_mask[0,:].unsqueeze(0)])
    
    Trainer.fit(train_loader, val_loader)
    Trainer.test(test_loader, model_location=os.path.join(config.training.log_dir, 'checkpoint.pt'))
    

if __name__ == '__main__':
    parser = ArgumentParser()

    parser.add_argument('--config', type=str, default='configs/config-train-pmhc-model.yml')
    # parser.add_argument('--config', type=str, default='configs/config-train-pmhc-model-baseline.yml')
    args = parser.parse_args()

    main(args)