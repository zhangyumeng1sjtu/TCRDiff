import time
import os
from tqdm import tqdm

import torch
import torch.nn as nn
import numpy as np
import pandas as pd
from scipy.stats import pearsonr
from sklearn.metrics import accuracy_score, f1_score, average_precision_score

from .base import BaseTrainer
from ..utils.tokenizer import PeptideTokenizer
from ..model import PeptideMHCPairFormer, PeptideMHCModel
from ..model.model_utils import EarlyStopping
from .loss import Coord2SeqCrossEntropyLoss


class PeptideMHCTrainer(BaseTrainer):
    
    def __init__(self, config, log_dir=None):
        super(PeptideMHCTrainer, self).__init__(config, log_dir)
        
        self.tokenizer = PeptideTokenizer()
        self.model = PeptideMHCPairFormer(**config.model, tokenizer=self.tokenizer)
        # self.model = PeptideMHCModel(**config.model, tokenizer=self.tokenizer)
        
        self.model.load_pretrained_peptide_model(config.training.pretrained_peptide_model)
        
        self.model.to(self.device)
        
        self.mlm_loss_fn = Coord2SeqCrossEntropyLoss(ignore_index=0)
        self.binding_loss_fn = nn.MSELoss() # binding affinity
        
        self.mlm_prob = config.training.mlm_prob
        self.subset_prob = config.training.subset_prob
        self.lambda_mlm = config.training.lambda_mlm
        
        # self.optimizer, self.scheduler = self.configure_optimizer(self.model.parameters(), config.training)
        self.early_stopping = EarlyStopping(patience=config.training.patience, checkpoint_dir=self.log_dir)
        
        param_finetune_list = []
        param_default_list = []

        # unfreeze the final transformer layer
        for param in self.model.peptide_model.parameters():
            param.requires_grad = True
            param_finetune_list.append(param)
            
        for name, param in self.model.named_parameters():
            if not name.startswith('peptide_model'):
                param_default_list.append(param)
        
        self.optimizer, self.scheduler = self.configure_optimizer([
            {'params': param_finetune_list, 'lr': config.training.lr / 10}, # / 5
            {'params': param_default_list, 'lr': config.training.lr},    
        ], config.training)
        
    @torch.no_grad()
    def inject_noise(self, tokens, subset_prob=0.2):
        
        padding_idx = self.tokenizer.padding_idx
        mask_idx = self.tokenizer.mask_idx

        def _mlm_mask(inputs):
            prev_tokens = inputs.clone()
            labels = inputs.clone()
            
            probability_matrix = torch.full(labels.shape, self.mlm_prob).to(inputs.device)
            special_tokens_mask = (
                prev_tokens.eq(padding_idx)  # & mask
                | prev_tokens.eq(self.tokenizer.cls_idx)
                | prev_tokens.eq(self.tokenizer.sep_idx)
                | prev_tokens.eq(self.tokenizer.eos_idx)
            )
            probability_matrix.masked_fill_(special_tokens_mask, value=0.0)
            
            # Randomly decide rows to be completely unmasked
            no_mask_rows = torch.rand(labels.size(0)) < 1 - subset_prob
            probability_matrix[no_mask_rows] = 0  # Set mask probability to 0 for these rows
        
            masked_indices = torch.bernoulli(probability_matrix).bool()
            
            # 80% of the time, we replace masked input tokens with tokenizer.mask_token ([MASK])
            indices_replaced = torch.bernoulli(torch.full_like(probability_matrix, 0.8)).bool() & masked_indices
            prev_tokens[indices_replaced] = mask_idx

            # 10% of the time, we replace masked input tokens with random word
            indices_random = torch.bernoulli(torch.full_like(probability_matrix, 0.5)).bool() & masked_indices & ~indices_replaced
            # Convert the list of standard token indices into a tensor for indexing
            std_token_idxes = torch.tensor(self.tokenizer.standard_token_idxes, dtype=torch.long, device=prev_tokens.device)

            # Sample random indices from the standard token set
            random_idxes = torch.randint(len(std_token_idxes), labels.shape, dtype=torch.long, device=prev_tokens.device)
            random_words = std_token_idxes[random_idxes]
            prev_tokens[indices_random] = random_words[indices_random]
            
            return prev_tokens, masked_indices

        prev_tokens, prev_tokens_mask = _mlm_mask(tokens)
        
        return prev_tokens, prev_tokens_mask
        
    def step(self, batch, mask_peptide=True):
        pep_toks = batch['peptide_token'].to(self.device)
        mhc_embeds = batch['mhc_embedding'].to(self.device)
        mhc_pseudo_mask = batch['mhc_pseudo_mask'].to(self.device)
        targets = batch['target'].to(self.device)
        
        if mask_peptide:
            noised_pep_toks, noise_pep_mask = self.inject_noise(pep_toks, self.subset_prob)
            out, logits = self.model(noised_pep_toks, mhc_embeds, mhc_pseudo_mask)
            mlm_loss, logging_output = self.mlm_loss_fn(logits, pep_toks, label_mask=noise_pep_mask)
        else:
            out, logits = self.model(pep_toks, mhc_embeds, mhc_pseudo_mask)
            mlm_loss, logging_output = self.mlm_loss_fn(logits, pep_toks)
        
        binding_loss = self.binding_loss_fn(out.squeeze(-1), targets)
        
        loss = binding_loss + self.lambda_mlm * mlm_loss
        logging_output['mlm_loss'] = mlm_loss.data
        logging_output['binding_loss'] = binding_loss.data
        logging_output['pred'] = out.squeeze(-1).detach().cpu().numpy()
        logging_output['target'] = targets.detach().cpu().numpy()
        
        return loss, logging_output
    
    def train_one_epoch(self, train_loader):
        avg_mlm_loss = 0
        avg_ppl = 0
        sample_size = 0 
        
        avg_loss = 0
        avg_binding_loss = 0
        data_size = 0
        
        pred = []
        y = []
        
        self.model.train()
        for batch in train_loader:
            self.optimizer.zero_grad()

            loss, logging_output = self.step(batch)

            num_samples = logging_output['sample_size'].cpu().item()
            avg_mlm_loss += logging_output['mlm_loss'].cpu().item() * num_samples
            avg_ppl += logging_output['ppl'].cpu().item() * num_samples
            sample_size += num_samples
            
            batch_size = logging_output['bsz']
            avg_loss += loss.cpu().item() * batch_size
            avg_binding_loss += logging_output['binding_loss'].cpu().item() * batch_size
            data_size += batch_size
            
            pred.append(logging_output['pred'])
            y.append(logging_output['target'])
            
            loss.backward()
            self.optimizer.step()
            
        pred = np.concatenate(pred)
        y = np.concatenate(y)

        return {
            'loss': avg_loss / data_size,
            'binding_loss': avg_binding_loss / data_size,
            'data_size': data_size,
            'mlm_loss': avg_mlm_loss / sample_size,
            'ppl': avg_ppl / sample_size,
            'sample_size': sample_size,
            'coef': pearsonr(pred, y).statistic
        }
           
    def evaluate_one_epoch(self, eval_loader, return_predictions=False, mask_peptide=False):
        avg_mlm_loss = 0
        avg_ppl = 0
        sample_size = 0 
        
        avg_loss = 0
        avg_binding_loss = 0
        data_size = 0
        
        pred = []
        y = []
        
        self.model.eval()
        with torch.no_grad():
            for batch in eval_loader:
                loss, logging_output = self.step(batch, mask_peptide)

                num_samples = logging_output['sample_size'].cpu().item()
                avg_mlm_loss += logging_output['mlm_loss'].cpu().item() * num_samples
                avg_ppl += logging_output['ppl'].cpu().item() * num_samples
                sample_size += num_samples
                
                batch_size = logging_output['bsz']
                avg_loss += loss.cpu().item() * batch_size
                avg_binding_loss += logging_output['binding_loss'].cpu().item() * batch_size
                data_size += batch_size
                
                pred.append(logging_output['pred'])
                y.append(logging_output['target'])
        
        pred = np.concatenate(pred)
        y = np.concatenate(y)
                
        res = {
            'loss': avg_loss / data_size,
            'binding_loss': avg_binding_loss / data_size,
            'data_size': data_size,
            'mlm_loss': avg_mlm_loss / sample_size,
            'ppl': avg_ppl / sample_size,
            'sample_size': sample_size,
            'coef': pearsonr(pred, y).statistic
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
            self.logger.info(f'Train Binding Loss: {train_metrics["binding_loss"]:.4f} | Coef: {train_metrics["coef"]:.3f} | MLM Loss: {train_metrics["mlm_loss"]:.3f}')
            self.logger.info(f'Valid Binding Loss: {val_metrics["binding_loss"]:.4f} | Coef: {val_metrics["coef"]:.3f} | MLM Loss: {val_metrics["mlm_loss"]:.3f}')
            
            for key in ['binding_loss', 'mlm_loss', 'coef']:
                self.writer.add_scalar(f'Train/{key}', train_metrics[key], epoch+1)
                self.writer.add_scalar(f'Valid/{key}', val_metrics[key], epoch+1)
                
            self.early_stopping(val_metrics["coef"], self.model, goal="maximize")
            if self.early_stopping.early_stop:
                self.logger.info(f"Early stopping at Epoch {epoch+1}")
                break
        
        self.writer.close()
    
    def test(self, test_loader, model_location):
        self.model.load_state_dict(torch.load(model_location, weights_only=True))
        
        test_res = self.evaluate_one_epoch(test_loader, return_predictions=True, mask_peptide=False)
        self.logger.info(f'Test Binding Loss: {test_res["binding_loss"]:.4f} | Coef: {test_res["coef"]:.3f}')

        res = pd.DataFrame({'Target': test_res['target'], 'Pred': test_res['pred']})
        res.to_csv(os.path.join(self.log_dir, 'test_result.csv'), index=False)

    def predict(self, data_loader, model_location):
        self.model.load_state_dict(torch.load(model_location, weights_only=True))
        pred = []
        self.model.eval()
        with torch.no_grad():
            for batch in tqdm(data_loader):
                peptide_tokens= batch['peptide_token'].to(self.device)
                mhc_embeds = batch['mhc_embedding'].to(self.device)
                mhc_pseudo_mask = batch['mhc_pseudo_mask'].to(self.device)
                
                # remove masks when evaluating
                out, _, _ = self.model(peptide_tokens, mhc_embeds, mhc_pseudo_mask)
                pred.append(out.squeeze(-1).detach().cpu().numpy())
                
        pred = np.concatenate(pred)
        df = pd.DataFrame({'Pred': pred})
        df.to_csv(os.path.join(self.log_dir, 'predictions.csv'), index=False)
        
        
class PeptideMHCELTrainer(BaseTrainer):
    def __init__(self, config, log_dir=None):
        
        super(PeptideMHCELTrainer, self).__init__(config, log_dir)
        self.tokenizer = PeptideTokenizer()
        self.model = PeptideMHCPairFormer(**config.model, tokenizer=self.tokenizer)
        # self.model = PeptideMHCModel(**config.model, tokenizer=self.tokenizer)
        
        # load pre-trained peptide LM
        self.model.load_pretrained_peptide_model(config.training.pretrained_peptide_model)
        self.model.to(self.device)
        
        self.mlm_loss_fn = Coord2SeqCrossEntropyLoss(ignore_index=0)
        self.binding_loss_fn = nn.BCELoss() # eluted ligands
        self.lambda_mlm = config.training.lambda_mlm

        self.mlm_prob = config.training.mlm_prob
        self.subset_prob = config.training.subset_prob
        
        # self.optimizer, self.scheduler = self.configure_optimizer(self.model.parameters(), config.training)
        self.early_stopping = EarlyStopping(patience=config.training.patience, checkpoint_dir=self.log_dir)
        
        param_finetune_list = []
        param_default_list = []

        for param in self.model.peptide_model.parameters():
        # for param in self.model.peptide_model.transformer[-1].parameters():
            param.requires_grad = True
            param_finetune_list.append(param)
            
        for name, param in self.model.named_parameters():
            if not name.startswith('peptide_model'):
                param_default_list.append(param)
        
        self.optimizer, self.scheduler = self.configure_optimizer([
            {'params': param_finetune_list, 'lr': config.training.lr / 10}, # / 5
            {'params': param_default_list, 'lr': config.training.lr},    
        ], config.training)

    @torch.no_grad()
    def inject_noise(self, tokens, subset_prob=0.2):
        
        padding_idx = self.tokenizer.padding_idx
        mask_idx = self.tokenizer.mask_idx

        def _mlm_mask(inputs):
            prev_tokens = inputs.clone()
            labels = inputs.clone()
            
            probability_matrix = torch.full(labels.shape, self.mlm_prob).to(inputs.device)
            special_tokens_mask = (
                prev_tokens.eq(padding_idx)  # & mask
                | prev_tokens.eq(self.tokenizer.cls_idx)
                | prev_tokens.eq(self.tokenizer.sep_idx)
                | prev_tokens.eq(self.tokenizer.eos_idx)
            )
            probability_matrix.masked_fill_(special_tokens_mask, value=0.0)
            
            # Randomly decide rows to be completely unmasked
            no_mask_rows = torch.rand(labels.size(0)) < 1 - subset_prob
            probability_matrix[no_mask_rows] = 0  # Set mask probability to 0 for these rows
        
            masked_indices = torch.bernoulli(probability_matrix).bool()
            
            # 80% of the time, we replace masked input tokens with tokenizer.mask_token ([MASK])
            indices_replaced = torch.bernoulli(torch.full_like(probability_matrix, 0.8)).bool() & masked_indices
            prev_tokens[indices_replaced] = mask_idx

            # 10% of the time, we replace masked input tokens with random word
            indices_random = torch.bernoulli(torch.full_like(probability_matrix, 0.5)).bool() & masked_indices & ~indices_replaced
            # Convert the list of standard token indices into a tensor for indexing
            std_token_idxes = torch.tensor(self.tokenizer.standard_token_idxes, dtype=torch.long, device=prev_tokens.device)

            # Sample random indices from the standard token set
            random_idxes = torch.randint(len(std_token_idxes), labels.shape, dtype=torch.long, device=prev_tokens.device)
            random_words = std_token_idxes[random_idxes]
            prev_tokens[indices_random] = random_words[indices_random]
            
            return prev_tokens, masked_indices

        prev_tokens, prev_tokens_mask = _mlm_mask(tokens)
        
        return prev_tokens, prev_tokens_mask
    
    def step(self, batch, mask_peptide=True):
        pep_toks = batch['peptide_token'].to(self.device)
        mhc_embeds = batch['mhc_embedding'].to(self.device)
        mhc_pseudo_mask = batch['mhc_pseudo_mask'].to(self.device)
        targets = batch['target'].to(self.device)
        
        if mask_peptide:
            noised_pep_toks, noise_pep_mask = self.inject_noise(pep_toks, self.subset_prob)
            out, logits = self.model(noised_pep_toks, mhc_embeds, mhc_pseudo_mask)
            mlm_loss, logging_output = self.mlm_loss_fn(logits, pep_toks, label_mask=noise_pep_mask)
        else:
            out, logits = self.model(pep_toks, mhc_embeds, mhc_pseudo_mask)
            mlm_loss, logging_output = self.mlm_loss_fn(logits, pep_toks)
        
        binding_loss = self.binding_loss_fn(out.squeeze(-1), targets)
        
        loss = binding_loss + self.lambda_mlm * mlm_loss
        logging_output['mlm_loss'] = mlm_loss.data
        logging_output['binding_loss'] = binding_loss.data
        logging_output['pred'] = out.squeeze(-1).detach().cpu().numpy()
        logging_output['target'] = targets.detach().cpu().numpy()
        
        return loss, logging_output
    
    def train_one_epoch(self, train_loader):
        avg_mlm_loss = 0
        avg_ppl = 0
        sample_size = 0 
        
        avg_loss = 0
        avg_binding_loss = 0
        data_size = 0
        
        pred = []
        y = []
        
        self.model.train()
        for batch in train_loader:
        # for batch in tqdm(train_loader):
            self.optimizer.zero_grad()

            loss, logging_output = self.step(batch)

            num_samples = logging_output['sample_size'].cpu().item()
            avg_mlm_loss += logging_output['mlm_loss'].cpu().item() * num_samples
            avg_ppl += logging_output['ppl'].cpu().item() * num_samples
            sample_size += num_samples
            
            batch_size = logging_output['bsz']
            avg_loss += loss.cpu().item() * batch_size
            avg_binding_loss += logging_output['binding_loss'].cpu().item() * batch_size
            data_size += batch_size
            
            pred.append(logging_output['pred'])
            y.append(logging_output['target'])
            
            loss.backward()
            self.optimizer.step()
            
        pred = np.concatenate(pred)
        y = np.concatenate(y)

        return {
            'loss': avg_loss / data_size,
            'binding_loss': avg_binding_loss / data_size,
            'data_size': data_size,
            'mlm_loss': avg_mlm_loss / sample_size,
            'ppl': avg_ppl / sample_size,
            'sample_size': sample_size,
            'acc': accuracy_score(y, (pred >= 0.5).astype(int)),
            'f1': f1_score(y, (pred >= 0.5).astype(int)),
            'aupr': average_precision_score(y, pred)
        }
           
    def evaluate_one_epoch(self, eval_loader, return_predictions=False, mask_peptide=False):
        avg_mlm_loss = 0
        avg_ppl = 0
        sample_size = 0 
        
        avg_loss = 0
        avg_binding_loss = 0
        data_size = 0
        
        pred = []
        y = []
        
        self.model.eval()
        with torch.no_grad():
            for batch in eval_loader:
            # for batch in tqdm(eval_loader):
                loss, logging_output = self.step(batch, mask_peptide)

                num_samples = logging_output['sample_size'].cpu().item()
                avg_mlm_loss += logging_output['mlm_loss'].cpu().item() * num_samples
                avg_ppl += logging_output['ppl'].cpu().item() * num_samples
                sample_size += num_samples
                
                batch_size = logging_output['bsz']
                avg_loss += loss.cpu().item() * batch_size
                avg_binding_loss += logging_output['binding_loss'].cpu().item() * batch_size
                data_size += batch_size
                
                pred.append(logging_output['pred'])
                y.append(logging_output['target'])
        
        pred = np.concatenate(pred)
        y = np.concatenate(y)
                
        res = {
            'loss': avg_loss / data_size,
            'binding_loss': avg_binding_loss / data_size,
            'data_size': data_size,
            'mlm_loss': avg_mlm_loss / sample_size,
            'ppl': avg_ppl / sample_size,
            'sample_size': sample_size,
            'acc': accuracy_score(y, (pred >= 0.5).astype(int)),
            'f1': f1_score(y, (pred >= 0.5).astype(int)),
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
            self.logger.info(f'Train Binding Loss: {train_metrics["binding_loss"]:.4f} | MLM Loss: {train_metrics["mlm_loss"]:.3f}')
            self.logger.info(f'Train ACC: {train_metrics["acc"]*100:.2f}% | F1: {train_metrics["f1"]:.3f} | AUPR: {train_metrics["aupr"]:.3f}')
            self.logger.info(f'Valid Binding Loss: {val_metrics["binding_loss"]:.4f} | MLM Loss: {val_metrics["mlm_loss"]:.3f}')
            self.logger.info(f'Valid ACC: {val_metrics["acc"]*100:.2f}% | F1: {val_metrics["f1"]:.3f} | AUPR: {val_metrics["aupr"]:.3f}')
            
            for key in ['binding_loss', 'mlm_loss', 'acc', 'f1', 'aupr']:
                self.writer.add_scalar(f'Train/{key}', train_metrics[key], epoch+1)
                self.writer.add_scalar(f'Valid/{key}', val_metrics[key], epoch+1)
                
            self.early_stopping(val_metrics["aupr"], self.model, goal="maximize")
            if self.early_stopping.early_stop:
                self.logger.info(f"Early stopping at Epoch {epoch+1}")
                break
        
        self.writer.close()
    
    def test(self, test_loader, model_location):
        self.model.load_state_dict(torch.load(model_location, weights_only=True))
        
        test_res = self.evaluate_one_epoch(test_loader, return_predictions=True, mask_peptide=False)
        self.logger.info(f'Test Binding Loss: {test_res["binding_loss"]:.4f}')
        self.logger.info(f'Test ACC: {test_res["acc"]*100:.2f}% | F1: {test_res["f1"]:.3f} | AUPR: {test_res["aupr"]:.3f}')

        res = pd.DataFrame({'Target': test_res['target'], 'Pred': test_res['pred']})
        res.to_csv(os.path.join(self.log_dir, 'test_result.csv'), index=False)

    def predict(self, data_loader, model_location):
        self.model.load_state_dict(torch.load(model_location, weights_only=True))
        pred = []
        self.model.eval()
        with torch.no_grad():
            for batch in tqdm(data_loader):
                peptide_tokens= batch['peptide_token'].to(self.device)
                mhc_embeds = batch['mhc_embedding'].to(self.device)
                mhc_pseudo_mask = batch['mhc_pseudo_mask'].to(self.device)
                
                # remove masks when evaluating
                out, _, _ = self.model(peptide_tokens, mhc_embeds, mhc_pseudo_mask)
                pred.append(out.squeeze(-1).detach().cpu().numpy())
                
        pred = np.concatenate(pred)
        df = pd.DataFrame({'Pred': pred})
        df.to_csv(os.path.join(self.log_dir, 'predictions.csv'), index=False)
        