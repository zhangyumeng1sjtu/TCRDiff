#TODO: TCR-pMHC binding prediction model trainer
# Use TCR-pMHC PairFormer model and need sample negative pairs from background data or internal "non-binding" tcrs
# Benchmarking on validation set, compare with EPACT-like contrastive model w/o PairFormer architecture

import time
import os
from tqdm import tqdm

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score, average_precision_score, accuracy_score

from .base import BaseTrainer
from ..utils.tokenizer import PairCDR3Tokenizer, PeptideTokenizer
from ..model import TCRpMHCPairFormer, TCRpMHCCoembeddingModel, TCRpMHCCoembeddingPairFormer
from ..model.model_utils import EarlyStopping


class TCRpMHCBindingTrainer(BaseTrainer):
    def __init__(self, config, log_dir=None):
        super(TCRpMHCBindingTrainer, self).__init__(config, log_dir)
        
        self.tcr_tokenizer = PairCDR3Tokenizer()
        self.pep_tokenizer = PeptideTokenizer()
        
        self.model = TCRpMHCPairFormer(config.model.tcr_model, config.model.pmhc_model, **config.model.pairformer)
        
        self.model.load_pretrained_weights(config.training.pretrained_tcr_model, config.training.pretrained_pmhc_model)
        self.model.to(self.device)
        
        self.loss_fn = nn.BCELoss()
        
        # unfreeze pretrained model parameter (using smaller lr) ?? maybe unfreeze the final layer?
        # param_finetune_list = []
        # param_default_list = []

        # for param in self.model.tcr_model.net.transformer[-1].parameters():
        #     param.requires_grad = True
        #     param_finetune_list.append(param)
            
        # for param in self.model.pmhc_model.pmhc_binding_pairformer.parameters():
        #     param.requires_grad = True
        #     param_finetune_list.append(param)

        # for name, param in self.model.named_parameters():
        #     if not name.startswith('tcr_model') and not name.startswith('pmhc_model'):
        #         param_default_list.append(param)
        
        # self.optimizer, self.scheduler = self.configure_optimizer([
        #     {'params': param_finetune_list, 'lr': config.training.lr / 10}, # / 20
        #     {'params': param_default_list, 'lr': config.training.lr},    
        # ], config.training)
        
        self.optimizer, self.scheduler = self.configure_optimizer(self.model.parameters(), config.training)
        self.early_stopping = EarlyStopping(patience=config.training.patience, checkpoint_dir=self.log_dir)
    
    def step(self, batch):
        for key, value in batch.items():
            if isinstance(value, torch.Tensor):
                batch[key] = value.to(self.device)
        out = self.model(batch)
        loss = self.loss_fn(out.squeeze(-1), batch['target'])

        logging_output = {
            'bsz': batch['target'].size(0),
            'pred': out.squeeze(-1).detach().cpu().numpy(),
            'target': batch['target'].detach().cpu().numpy()
        }
        return loss, logging_output

    def train_one_epoch(self, train_loader):
        avg_loss = 0
        data_size = 0

        pred = []
        y = []
        
        self.model.train()
        for batch in train_loader:
            self.optimizer.zero_grad()

            loss, logging_output = self.step(batch)
            
            batch_size = logging_output['bsz']
            avg_loss += loss.cpu().item() * batch_size
            data_size += batch_size
            
            pred.append(logging_output['pred'])
            y.append(logging_output['target'])
            
            loss.backward()
            self.optimizer.step()
            
        pred = np.concatenate(pred)
        y = np.concatenate(y)

        return {
            'loss': avg_loss / data_size,
            'data_size': data_size,
            'acc': accuracy_score(y, (pred >= 0.5).astype(int)),
            'auc': roc_auc_score(y, pred),
            'auc0.1': roc_auc_score(y, pred, max_fpr=0.1),
            'aupr': average_precision_score(y, pred)
        }
    
    def evaluate_one_epoch(self, eval_loader, return_predictions=False):
        avg_loss = 0
        data_size = 0
        
        pred = []
        y = []
        
        self.model.eval()
        with torch.no_grad():
            for batch in eval_loader:
                loss, logging_output = self.step(batch)
                
                batch_size = logging_output['bsz']
                avg_loss += loss.cpu().item() * batch_size
                data_size += batch_size
                
                pred.append(logging_output['pred'])
                y.append(logging_output['target'])
        
        pred = np.concatenate(pred)
        y = np.concatenate(y)
                
        res = {
            'loss': avg_loss / data_size,
            'data_size': data_size,
            'acc': accuracy_score(y, (pred >= 0.5).astype(int)),
            'auc': roc_auc_score(y, pred),
            'auc0.1': roc_auc_score(y, pred, max_fpr=0.1),
            'aupr': average_precision_score(y, pred)
        }
        
        if return_predictions:
            res['pred'] = pred
            res['target'] = y
        
        return res
    
    def fit(self, train_loader, val_loader):
        self.configure_logger(self.log_dir)

        for epoch in range(self.max_epochs):
            start_time = time.time()

            train_metrics= self.train_one_epoch(train_loader)
            val_metrics = self.evaluate_one_epoch(val_loader)
            self.scheduler.step()

            end_time = time.time()
            epoch_secs = end_time - start_time
            
            self.logger.info(f'Epoch: {epoch+1:02} | Epoch Time: {epoch_secs:.2f}s')
            self.logger.info(f'Train Loss: {train_metrics["loss"]:.4f} | ACC: {train_metrics["acc"]*100:.2f}% | AUC: {train_metrics["auc"]:.3f} | AUC0.1: {train_metrics["auc0.1"]:.3f} | AUPR: {train_metrics["aupr"]:.3f}')
            self.logger.info(f'Valid Loss: {val_metrics["loss"]:.4f} | ACC: {val_metrics["acc"]*100:.2f}% | AUC: {val_metrics["auc"]:.3f} | AUC0.1: {val_metrics["auc0.1"]:.3f} | AUPR: {val_metrics["aupr"]:.3f}')
            
            for key in ['loss', 'acc', 'auc', 'auc0.1', 'aupr']:
                self.writer.add_scalar(f'Train/{key}', train_metrics[key], epoch+1)
                self.writer.add_scalar(f'Valid/{key}', val_metrics[key], epoch+1)
                
            self.early_stopping(val_metrics["aupr"], self.model, goal="maximize")
            # self.early_stopping(val_metrics["auc0.1"], self.model, goal="maximize")
            
            if self.early_stopping.early_stop:
                self.logger.info(f"Early stopping at Epoch {epoch+1}")
                break
        
        self.writer.close()
        
    def test(self, test_loader, model_location):
        self.model.load_state_dict(torch.load(model_location, map_location='cpu', weights_only=True))
        
        test_res = self.evaluate_one_epoch(test_loader, return_predictions=True)
        self.logger.info(f'Test Loss: {test_res["loss"]:.4f} | ACC: {test_res["acc"]*100:.2f}% | AUC: {test_res["auc"]:.3f} | AUC0.1: {test_res["auc0.1"]:.3f} | AUPR: {test_res["aupr"]:.3f}')

        res = pd.DataFrame({'Target': test_res['target'], 'Pred': test_res['pred']})
        res.to_csv(os.path.join(self.log_dir, 'test_result.csv'), index=False)
        
    def predict(self, data_loader, model_location, out_dir=None):
        self.model.load_state_dict(torch.load(model_location, weights_only=True, map_location='cpu'))
        pred = []
        self.model.eval()
        with torch.no_grad():
            for batch in tqdm(data_loader):
                for key, value in batch.items():
                    if isinstance(value, torch.Tensor):
                        batch[key] = value.to(self.device)
                        
                out= self.model(batch)
                pred.append(out.squeeze(-1).detach().cpu().numpy())
                
        pred = np.concatenate(pred)
        df = pd.DataFrame({'Pred': pred})
        
        out_dir = self.log_dir if out_dir is None else out_dir
        df.to_csv(os.path.join(out_dir, 'predictions.csv'), index=False)
        

class TCRpMHCCoembeddingTrainer(BaseTrainer):
    
    def __init__(self, config, log_dir=None):
        super(TCRpMHCCoembeddingTrainer, self).__init__(config, log_dir)
        
        self.tcr_tokenizer = PairCDR3Tokenizer()
        self.pep_tokenizer = PeptideTokenizer()
        
        self.model = TCRpMHCCoembeddingModel(config.model.tcr_model, config.model.pmhc_model, **config.model.coembedding)
        # self.model = TCRpMHCCoembeddingPairFormer(config.model.tcr_model, config.model.pmhc_model, **config.model.pairformer)
        
        self.model.load_pretrained_weights(config.training.pretrained_tcr_model, config.training.pretrained_pmhc_model)
        self.model.to(self.device)
        
        self.optimizer, self.scheduler = self.configure_optimizer(self.model.parameters(), config.training)
        
        # unfreeze pretrained model parameter (using smaller lr) ?? maybe unfreeze the final layer?
        # param_finetune_list = []
        # param_default_list = []

        # for param in self.model.tcr_model.net.transformer[-1].parameters():
        #     param.requires_grad = True
        #     param_finetune_list.append(param)
            
        # for param in self.model.pmhc_model.pmhc_binding_pairformer.parameters():
        #     param.requires_grad = True
        #     param_finetune_list.append(param)
        
        # for param in self.model.pmhc_model.peptide_cross_attn.parameters():
        #     param.requires_grad = True
        #     param_finetune_list.append(param)
        # for param in self.model.pmhc_model.mhc_cross_attn.parameters():
        #     param.requires_grad = True
        #     param_finetune_list.append(param)
            
        # for name, param in self.model.named_parameters():
        #     if not name.startswith('tcr_model') and not name.startswith('pmhc_model'):
        #         param_default_list.append(param)
        
        # self.optimizer, self.scheduler = self.configure_optimizer([
        #     {'params': param_finetune_list, 'lr': config.training.lr / 10}, # / 10
        #     {'params': param_default_list, 'lr': config.training.lr},    
        # ], config.training)
        
        self.early_stopping = EarlyStopping(patience=config.training.patience, checkpoint_dir=self.log_dir)
        
        self.temperature = config.training.temperature
        self.contrastive_loss_coef = config.training.contrastive_loss_coef
        
    @staticmethod
    def compute_contrastive_loss(anchor, positive, negative, temperature=1.0):
        """
            anchor: B, E; positive B, E; negative: 5B, E
        """
        logits_pos = F.cosine_similarity(anchor, positive) # B
        negative = negative.view(-1, 5, anchor.size(1)) # B, 5, E
        logits_neg = torch.stack([F.cosine_similarity(anchor, neg) for neg in negative.transpose(0, 1)], dim=1) # B, 5
        
        numerator = torch.exp(logits_pos / temperature)
        denominator = numerator + torch.exp(logits_neg / temperature).sum(1)
        
        loss = - torch.log(numerator / denominator)
        return loss.mean()
    
    def step(self, batch, training=True):
        for key, value in batch.items():
            if isinstance(value, torch.Tensor):
                batch[key] = value.to(self.device)
                
        out = self.model(batch)
        logits = out['logits'].squeeze(-1)
        labels = batch['target']
        
        if training:
            tcr_projections, pmhc_projections = out['projection']
            
            pos_batch_size = labels.shape[0] // 6
            anchor_projections = pmhc_projections[:pos_batch_size, :]
            positive_projections = tcr_projections[:pos_batch_size, :]
            negative_projections = tcr_projections[pos_batch_size:, :]
            
            bce_loss = F.binary_cross_entropy(logits, labels)
            contrastive_loss = self.compute_contrastive_loss(anchor_projections, positive_projections, negative_projections, temperature=self.temperature)
                
            loss = bce_loss + self.contrastive_loss_coef * contrastive_loss
            logging_output = {
                'bce_loss': bce_loss.detach().cpu().numpy(),
                'contrastive_loss': contrastive_loss.detach().cpu().numpy(),
                'bsz': labels.shape[0],
                'pred': logits.detach().cpu().numpy(),
                'target': labels.detach().cpu().numpy()
            }
            
        else:
            loss = F.binary_cross_entropy(logits, labels)
            logging_output = {
                'bce_loss': loss.detach().cpu().numpy(),
                'bsz': labels.shape[0],
                'pred': logits.detach().cpu().numpy(),
                'target': labels.detach().cpu().numpy()
            }
        
        return loss, logging_output
    
    def train_one_epoch(self, train_loader):
        avg_loss = 0
        avg_bce_loss = 0
        avg_contrastive_loss = 0
        data_size = 0

        pred = []
        y = []
        
        self.model.train()
        for batch in train_loader:
            self.optimizer.zero_grad()

            loss, logging_output = self.step(batch, training=True)
            
            batch_size = logging_output['bsz']
            avg_loss += loss.cpu().item() * batch_size
            avg_bce_loss += logging_output['bce_loss'] * batch_size
            avg_contrastive_loss += logging_output['contrastive_loss'] * batch_size
            data_size += batch_size
            
            pred.append(logging_output['pred'])
            y.append(logging_output['target'])
            
            loss.backward()
            self.optimizer.step()
            
        pred = np.concatenate(pred)
        y = np.concatenate(y)

        return {
            'loss': avg_loss / data_size,
            'bce_loss': avg_bce_loss / data_size,
            'contrastive_loss': avg_contrastive_loss / data_size,
            'data_size': data_size,
            'acc': accuracy_score(y, (pred >= 0.5).astype(int)),
            'auc': roc_auc_score(y, pred),
            'auc0.1': roc_auc_score(y, pred, max_fpr=0.1),
            'aupr': average_precision_score(y, pred)
        }
    
    def evaluate_one_epoch(self, eval_loader, return_predictions=False):
        avg_loss = 0
        data_size = 0
        
        pred = []
        y = []
        
        self.model.eval()
        with torch.no_grad():
            for batch in eval_loader:
                loss, logging_output = self.step(batch, training=False)
                
                batch_size = logging_output['bsz']
                avg_loss += loss.cpu().item() * batch_size
                data_size += batch_size
                
                pred.append(logging_output['pred'])
                y.append(logging_output['target'])
        
        pred = np.concatenate(pred)
        y = np.concatenate(y)
                
        res = {
            'loss': avg_loss / data_size,
            'data_size': data_size,
            'acc': accuracy_score(y, (pred >= 0.5).astype(int)),
            'auc': roc_auc_score(y, pred),
            'auc0.1': roc_auc_score(y, pred, max_fpr=0.1),
            'aupr': average_precision_score(y, pred)
        }
        
        if return_predictions:
            res['pred'] = pred
            res['target'] = y
        
        return res
    
    def fit(self, train_loader, val_loader):
        self.configure_logger(self.log_dir)

        for epoch in range(self.max_epochs):
            start_time = time.time()

            train_metrics= self.train_one_epoch(train_loader)
            val_metrics = self.evaluate_one_epoch(val_loader)
            self.scheduler.step()

            end_time = time.time()
            epoch_secs = end_time - start_time
            
            self.logger.info(f'Epoch: {epoch+1:02} | Epoch Time: {epoch_secs:.2f}s')
            self.logger.info(f'Train Loss: {train_metrics["loss"]:.4f} | BCE Loss: {train_metrics["bce_loss"]:.4f} | Contrastive Loss: {train_metrics["contrastive_loss"]:.4f}')
            self.logger.info(f'Train ACC: {train_metrics["acc"]*100:.2f}% | AUC: {train_metrics["auc"]:.3f} | AUC0.1: {train_metrics["auc0.1"]:.3f} | AUPR: {train_metrics["aupr"]:.3f}')
            self.logger.info(f'Valid Loss: {val_metrics["loss"]:.4f} | ACC: {val_metrics["acc"]*100:.2f}% | AUC: {val_metrics["auc"]:.3f} | AUC0.1: {val_metrics["auc0.1"]:.3f} | AUPR: {val_metrics["aupr"]:.3f}')
            
            for key in ['loss', 'acc', 'auc', 'auc0.1', 'aupr']:
                self.writer.add_scalar(f'Train/{key}', train_metrics[key], epoch+1)
                self.writer.add_scalar(f'Valid/{key}', val_metrics[key], epoch+1)
                
            self.early_stopping(val_metrics["aupr"], self.model, goal="maximize")
            
            if self.early_stopping.early_stop:
                self.logger.info(f"Early stopping at Epoch {epoch+1}")
                break
        
        self.writer.close()
        
    def test(self, test_loader, model_location):
        self.model.load_state_dict(torch.load(model_location, map_location='cpu', weights_only=True))
        
        test_res = self.evaluate_one_epoch(test_loader, return_predictions=True)
        self.logger.info(f'Test Loss: {test_res["loss"]:.4f} | ACC: {test_res["acc"]*100:.2f}% | AUC: {test_res["auc"]:.3f} | AUC0.1: {test_res["auc0.1"]:.3f} | AUPR: {test_res["aupr"]:.3f}')

        res = pd.DataFrame({'Target': test_res['target'], 'Pred': test_res['pred']})
        res.to_csv(os.path.join(self.log_dir, 'test_result.csv'), index=False)
        
    def predict(self, data_loader, model_location, out_dir=None):
        self.model.load_state_dict(torch.load(model_location, map_location='cpu', weights_only=True))
        pred = []
        self.model.eval()
        with torch.no_grad():
            for batch in tqdm(data_loader):
                for key, value in batch.items():
                    if isinstance(value, torch.Tensor):
                        batch[key] = value.to(self.device)
                        
                out = self.model(batch)
                pred.append(out['logits'].squeeze(-1).detach().cpu().numpy())
                
        pred = np.concatenate(pred)
        df = pd.DataFrame({'Pred': pred})
        
        out_dir = self.log_dir if out_dir is None else out_dir
        df.to_csv(os.path.join(out_dir, 'predictions.csv'), index=False)
        