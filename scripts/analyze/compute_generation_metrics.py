import argparse

from torch.utils.data import DataLoader

from TCRDiff.utils import set_seed, load_config
from TCRDiff.dataset import TCRpMHCDataset, TCRpMHCBatchConverter, TCRDataset, TCRBatchConverter
from TCRDiff.trainer import ConditionalTCRDPLMTrainer, TCRDPLMTrainer


parser = argparse.ArgumentParser()
parser.add_argument('--dirname', type=str, required=True)
parser.add_argument('--gpu_device', type=int, default=0)
args = parser.parse_args()

config_path = f"{args.dirname}/config.yml"
model_location = f"{args.dirname}/checkpoint.pt"

config = load_config(config_path)
set_seed(config.training.seed)
device = torch.device(f"cuda:{args.gpu_device}" if torch.cuda.is_available() else "cpu")
config.training.gpu_device = device

val_data = TCRpMHCDataset(data_path="data/tcrpmhc/validation_pos_data.csv",
                           mhc_lib_path="data/pmhc/mhc_lib.json",
                           v_gene_lib_path="data/tcrpmhc/v_gene_lib.csv",
                           use_cdr12_columns=True
)

test_data = TCRpMHCDataset(data_path="data/tcrpmhc/subsample/immrep23_test_pos_data.csv",
                           mhc_lib_path="data/pmhc/mhc_lib.json",
                           v_gene_lib_path="data/tcrpmhc/v_gene_lib.csv",
                           use_cdr12_columns=True
)

batch_converter = TCRpMHCBatchConverter(max_cdr3_len=config.data.max_cdr3_len, max_peptide_len=config.data.max_peptide_len, max_mhc_len=config.data.max_mhc_len) 

val_loader = DataLoader(val_data, batch_size=config.training.batch_size, collate_fn=batch_converter, num_workers=config.training.num_workers, shuffle=False)
test_loader = DataLoader(test_data, batch_size=config.training.batch_size, collate_fn=batch_converter, num_workers=config.training.num_workers, shuffle=False)

Trainer = ConditionalTCRDPLMTrainer(config) # conditioned on pMHC binding
Trainer.test(val_loader, model_location=model_location)
Trainer.test(test_loader, model_location=model_location)

val_pred = Trainer.predict(val_loader, model_location=model_location, noise = 'full_mask')
test_pred = Trainer.predict(test_loader, model_location=model_location, noise = 'full_mask')

print(f"Val ACC: {val_pred['accuracy']:.3f} | Alpha ACC: {val_pred['alpha_accuracy']:.3f} | Beta ACC: {val_pred['beta_accuracy']:.3f}")
print(f"Val Median ACC: {val_pred['median_accuracy']:.3f} | Alpha Median ACC: {val_pred['median_alpha_accuracy']:.3f} | Beta Median ACC: {val_pred['median_beta_accuracy']:.3f}")

print(f"Test ACC: {test_pred['accuracy']:.3f} | Alpha ACC: {test_pred['alpha_accuracy']:.3f} | Beta ACC: {test_pred['beta_accuracy']:.3f}")
print(f"Test Median ACC: {test_pred['median_accuracy']:.3f} | Alpha Median ACC: {test_pred['median_alpha_accuracy']:.3f} | Beta Median ACC: {test_pred['median_beta_accuracy']:.3f}")

# class-specific predictions
# class i and class ii validation data
val_data_class_i = TCRpMHCDataset(data_path="data/tcrpmhc/class_i/validation_pos_data_class_i.csv", 
                            mhc_lib_path="data/pmhc/mhc_lib.json",
                            v_gene_lib_path="data/tcrpmhc/v_gene_lib.csv",
                            use_cdr12_columns=True
)
val_data_class_ii = TCRpMHCDataset(data_path="data/tcrpmhc/class_ii/validation_pos_data_class_ii.csv",
                            mhc_lib_path="data/pmhc/mhc_lib.json",
                            v_gene_lib_path="data/tcrpmhc/v_gene_lib.csv",
                            use_cdr12_columns=True
)

val_loader_class_i = DataLoader(val_data_class_i, batch_size=config.training.batch_size, collate_fn=batch_converter, num_workers=config.training.num_workers, shuffle=False)
val_loader_class_ii = DataLoader(val_data_class_ii, batch_size=config.training.batch_size, collate_fn=batch_converter, num_workers=config.training.num_workers, shuffle=False)

val_pred_class_i = Trainer.predict(val_loader_class_i, model_location=model_location, noise = 'full_mask')
val_pred_class_ii = Trainer.predict(val_loader_class_ii, model_location=model_location, noise = 'full_mask')

print(f"Class I Val ACC: {val_pred_class_i['accuracy']:.3f} | Alpha ACC: {val_pred_class_i['alpha_accuracy']:.3f} | Beta ACC: {val_pred_class_i['beta_accuracy']:.3f}")
print(f"Class I Val Median ACC: {val_pred_class_i['median_accuracy']:.3f} | Alpha Median ACC: {val_pred_class_i['median_alpha_accuracy']:.3f} | Beta Median ACC: {val_pred_class_i['median_beta_accuracy']:.3f}")
print(f"Class II Val ACC: {val_pred_class_ii['accuracy']:.3f} | Alpha ACC: {val_pred_class_ii['alpha_accuracy']:.3f} | Beta ACC: {val_pred_class_ii['beta_accuracy']:.3f}")
print(f"Class II Val Median ACC: {val_pred_class_ii['median_accuracy']:.3f} | Alpha Median ACC: {val_pred_class_ii['median_alpha_accuracy']:.3f} | Beta Median ACC: {val_pred_class_ii['median_beta_accuracy']:.3f}")
