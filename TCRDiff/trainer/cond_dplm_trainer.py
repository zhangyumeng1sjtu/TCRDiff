import time

import torch
from tqdm import tqdm

from .base import BaseTrainer
from ..utils.tokenizer import PairCDR3Tokenizer
from ..model.cond_dplm import ConditionalDPLM
from ..model.model_utils import EarlyStopping
from .loss import RDMCrossEntropyLoss


class ConditionalTCRDPLMTrainer(BaseTrainer):
    
    def __init__(self, config):
        super(ConditionalTCRDPLMTrainer, self).__init__(config)
        
        self.tokenizer = PairCDR3Tokenizer()
        
        self.model = ConditionalDPLM(config.model.tcr_model, config.model.pmhc_model, config.model.adapter, config.training.num_diffusion_steps, config.model.adapter.dim_pairwise, config.model.adapter.dropout, config.training.use_pmhc_struc_feat)
        
        if config.training.pretrained_model is not None:
            # self.model.load_pretrained_weights(config.training.pretrained_pmhc_model, config.training.pretrained_tcr_model)
            self.model.load_pretrained_weights(config.training.pretrained_model) # load pretrained weights
            
        self.model.to(self.device)
        
        # configure criterion
        self.loss_fn = RDMCrossEntropyLoss(ignore_index=0) # ignore padding index
        self.timestep_weighting = config.training.timestep_weighting
        self.num_diffusion_steps = config.training.num_diffusion_steps
        
        self.max_iter = config.training.max_iter
        self.temperature = config.training.temperature
        self.sampling_strategy = config.training.sampling_strategy
        self.partial_chain_prob = config.training.partial_chain_prob
        self.partial_cdr12_prob = config.training.partial_cdr12_prob if config.training.partial_cdr12_prob is not None else 0.0
        
        # self.optimizer, self.scheduler = self.configure_optimizer(self.model.parameters(), config.training)
        
        # unfreeze the top tcr transformer layer
        param_finetune_list = []
        param_default_list = []
        for name, param in self.model.named_parameters():
            if name.startswith('decoder.tcr_model'):
                param_finetune_list.append(param)
            elif name.startswith('encoder'):
                param_finetune_list.append(param)
            else:
                param_default_list.append(param)
        
        self.optimizer, self.scheduler = self.configure_optimizer([
            {'params': param_finetune_list, 'lr': config.training.lr / 10}, # / 10
            {'params': param_default_list, 'lr': config.training.lr},    
        ], config.training)
        
        self.early_stopping = EarlyStopping(patience=config.training.patience, checkpoint_dir=self.log_dir)
    
    @torch.no_grad()
    def mask_partial_chain(self, chain_token_mask, mask_prob=0.2):
        num_samples, seq_len = chain_token_mask.size()
        
        # First, determine which samples have both chains
        has_alpha = (chain_token_mask == 1).any(dim=1)  # B,
        has_beta = (chain_token_mask == 2).any(dim=1)   # B,
        has_both_chains = has_alpha & has_beta          # B,
        
        # Only consider masking samples that have both chains
        eligible_samples = torch.zeros(num_samples, dtype=torch.bool, device=chain_token_mask.device)
        eligible_samples[has_both_chains] = torch.bernoulli(
            torch.full((has_both_chains.sum(),), mask_prob, device=chain_token_mask.device)
        ).bool()
        
        # For samples with both chains, randomly choose which chain to mask
        selected_chains = torch.zeros(num_samples, dtype=torch.long, device=chain_token_mask.device)
        # can use fixed chain selection by setting random_chain_selection to 1 or 2
        random_chain_selection = torch.randint(1, 3, size=(eligible_samples.sum(),), device=chain_token_mask.device) # either 1 or 2
        selected_chains[eligible_samples] = random_chain_selection
        
        # Expand sample mask to token level
        masked_samples_expanded = eligible_samples.unsqueeze(-1).expand(num_samples, seq_len)  # B, L
        chosen_chain_mask = (chain_token_mask == selected_chains.unsqueeze(-1))  # [B, L]

        # Return the final mask and the chain identifiers that were masked
        masked_chains = torch.where(eligible_samples, selected_chains, torch.zeros_like(selected_chains))
        
        return masked_samples_expanded & chosen_chain_mask, masked_chains
    
    def mask_partial_cdr12_feature(self, cdr12_alpha_feat, cdr12_beta_feat, mask_prob=0.2):
        B, L, D = cdr12_alpha_feat.size()
        mask = torch.bernoulli(torch.full((B,), mask_prob, device=cdr12_alpha_feat.device)).bool()

        masked_cdr12_alpha_feat = cdr12_alpha_feat.clone()
        masked_cdr12_alpha_feat[mask] = 0
        masked_cdr12_beta_feat = cdr12_beta_feat.clone()
        masked_cdr12_beta_feat[mask] = 0
        
        return masked_cdr12_alpha_feat, masked_cdr12_beta_feat, mask
    
    def step(self, batch):
        if self.partial_chain_prob > 0:
            partial_masks, _ = self.mask_partial_chain(batch['chain_token_mask'], mask_prob=self.partial_chain_prob)
        else:
            partial_masks = None

        if self.partial_cdr12_prob > 0:
            batch['cdr12_alpha_feat'], batch['cdr12_beta_feat'], _ = self.mask_partial_cdr12_feature(
                batch['cdr12_alpha_feat'], batch['cdr12_beta_feat'], mask_prob=self.partial_cdr12_prob
            )
        
        logits, target, loss_mask, weights = self.model(batch, weighting=self.timestep_weighting, partial_masks=partial_masks)
        
        # use entropy weights
        # updated_weights = batch['entropy_weights'].repeat(2,1) * weights
        
        loss, logging_output = self.loss_fn(
            logits, target, loss_mask, weights,
        )
        return loss, logging_output
        
    def train_one_epoch(self, train_loader):
        avg_loss = 0
        avg_nll_loss = 0
        avg_ppl = 0
        avg_acc = 0
        sample_size = 0 
        
        self.model.train()
        for batch in train_loader:
            self.optimizer.zero_grad()
            for key, value in batch.items():
                batch[key] = value.to(self.device)

            loss, logging_output = self.step(batch)

            num_samples = logging_output['sample_size'].cpu().item()
            avg_loss += loss.cpu().item() * num_samples
            avg_nll_loss += logging_output['nll_loss'].cpu().item() * num_samples
            avg_ppl += logging_output['ppl'].cpu().item() * num_samples
            avg_acc += logging_output['acc'].cpu().item() * num_samples
            sample_size += num_samples
            
            loss.backward()
            self.optimizer.step()
        
        return {
            'loss': avg_loss / sample_size,
            'nll_loss': avg_nll_loss / sample_size,
            'ppl': avg_ppl / sample_size,
            'acc': avg_acc / sample_size,
            'sample_size': sample_size
        }
        
    def evaluate_one_epoch(self, eval_loader):
        avg_loss = 0
        avg_nll_loss = 0
        avg_ppl = 0
        avg_acc = 0
        sample_size = 0
        
        self.model.eval()
        with torch.no_grad():
            for batch in eval_loader:
                for key, value in batch.items():
                    batch[key] = value.to(self.device)
                
                loss, logging_output = self.step(batch)

                num_samples = logging_output['sample_size'].cpu().item()
                avg_loss += loss.cpu().item() * num_samples
                avg_nll_loss += logging_output['nll_loss'].cpu().item() * num_samples
                avg_ppl += logging_output['ppl'].cpu().item() * num_samples
                avg_acc += logging_output['acc'].cpu().item() * num_samples
                sample_size += num_samples
                
        return {
            'loss': avg_loss / sample_size,
            'nll_loss': avg_nll_loss / sample_size,
            'ppl': avg_ppl / sample_size,
            'acc': avg_acc / sample_size,
            'sample_size': sample_size
        }
                
    
    def fit(self, train_loader, val_loader):
        self.configure_logger(self.log_dir)

        for epoch in range(self.max_epochs):
            start_time = time.time()

            train_res = self.train_one_epoch(train_loader)
            val_res = self.evaluate_one_epoch(val_loader)
            self.scheduler.step()

            end_time = time.time()
            epoch_secs = end_time - start_time

            self.logger.info(f'Epoch: {epoch+1:02} | Epoch Time: {epoch_secs:.2f}s')
            self.logger.info(f'Train Loss: {train_res["loss"]:.3f} | NLL Loss: {train_res["nll_loss"]:.3f} | Perplexity: {train_res["ppl"]:.2f} | ACC: {train_res["acc"]*100:.2f}%')
            self.logger.info(f'Valid Loss: {val_res["loss"]:.3f} | NLL Loss: {val_res["nll_loss"]:.3f} | Perplexity: {val_res["ppl"]:.2f} | ACC: {val_res["acc"]*100:.2f}%')

            self.writer.add_scalar('Train/loss', train_res['loss'], epoch+1)
            self.writer.add_scalar('Train/nll_loss', train_res['nll_loss'], epoch+1)
            self.writer.add_scalar('Train/ppl', train_res['ppl'], epoch+1)
            self.writer.add_scalar('Train/acc', train_res['acc'], epoch+1)
            
            self.writer.add_scalar('Valid/loss', val_res['loss'], epoch+1)
            self.writer.add_scalar('Valid/nll_loss', val_res['nll_loss'], epoch+1)
            self.writer.add_scalar('Valid/ppl', val_res['ppl'], epoch+1)
            self.writer.add_scalar('Valid/acc', val_res['acc'], epoch+1)
            
            val_pred = self.predict(val_loader, noise = 'full_mask')
            self.logger.info(f'Valid Generation ACC: {val_pred["accuracy"]:.3f} | Median ACC: {val_pred["median_accuracy"]:.3f}')

            # self.early_stopping(val_res['ppl'], self.model, goal="minimize")
            # self.early_stopping(val_res['acc'], self.model, goal="maximize")
            self.early_stopping(val_pred['accuracy'], self.model, goal="maximize")
            
            if self.early_stopping.early_stop:
                self.logger.info(f"Early stopping at Epoch {epoch+1}")
                break

        self.writer.close()

    def test(self, test_loader, model_location):
        self.model.load_state_dict(torch.load(model_location, weights_only=True, map_location='cpu'))
        test_res = self.evaluate_one_epoch(test_loader)
        # self.logger.info(f'Test Loss: {test_res['loss']:.3f} | NLL Loss: {test_res['nll_loss']:.3f} | Perplexity: {test_res['ppl']:.2f} | ACC: {test_res['acc']*100:.2f}%')
        print(f'Test Loss: {test_res["loss"]:.3f} | NLL Loss: {test_res["nll_loss"]:.3f} | Perplexity: {test_res["ppl"]:.2f} | ACC: {test_res["acc"]*100:.2f}%')
        
    # Inference/Generation
    def forward(self, batch, max_iter, sampling_strategy, temperature=1.0, partial_masks = None, use_draft_seq=False, decode_seq=False):
        output_tokens, output_scores = self.model.generate(
            batch = batch,
            max_iter = max_iter,
            sampling_strategy = sampling_strategy,
            temperature = temperature,
            partial_masks = partial_masks,
            use_draft_seq = use_draft_seq,
        )
        if decode_seq:
            return [''.join(seq.split(' ')) for seq in self.tokenizer.batch_decode(output_tokens, remove_special_tokens=True)]
        
        return output_tokens, output_scores
    
    @torch.no_grad()
    def inject_noise(self, tokens, noise=None, sel_mask=None):
        padding_idx = self.tokenizer.padding_idx
        mask_idx = self.tokenizer.mask_idx
        
        def new_arange(x, *size):
            """
            Return a Tensor of `size` filled with a range function on the device of x.
            If size is empty, using the size of the variable x.
            """
            if len(size) == 0:
                size = x.size()
            return torch.arange(size[-1], device=x.device).expand(*size).contiguous()

        def _full_mask(target_tokens):
            target_mask = (
                target_tokens.ne(padding_idx)  # & mask
                & target_tokens.ne(self.tokenizer.cls_idx)
                & target_tokens.ne(self.tokenizer.eos_idx)
                & target_tokens.ne(self.tokenizer.sep_idx)
            )
            masked_target_tokens = target_tokens.masked_fill(target_mask, mask_idx)
            return masked_target_tokens

        def _random_mask(target_tokens):
            target_masks = (
                target_tokens.ne(padding_idx)
                & target_tokens.ne(self.tokenizer.cls_idx)
                & target_tokens.ne(self.tokenizer.eos_idx)
                & target_tokens.ne(self.tokenizer.sep_idx)
            )
            target_score = target_tokens.clone().float().uniform_()
            target_score.masked_fill_(~target_masks, 2.0)
            
            target_length = target_masks.sum(1).float()
            target_length = target_length * target_length.clone().uniform_()
            target_length = target_length + 1  # make sure to mask at least one token.

            _, target_rank = target_score.sort(1)
            target_cutoff = new_arange(target_rank) < target_length[:, None].long()
            masked_target_tokens = target_tokens.masked_fill(
                target_cutoff.scatter(1, target_rank, target_cutoff), mask_idx
            )
            return masked_target_tokens 

        def _selected_mask(target_tokens, sel_mask):
            masked_target_tokens = torch.masked_fill(target_tokens, mask=sel_mask, value=mask_idx)
            return masked_target_tokens

        if noise == 'full_mask':
            masked_tokens = _full_mask(tokens)
        elif noise == 'random_mask':
            masked_tokens = _random_mask(tokens)
        elif noise == 'selected_mask':
            masked_tokens = _selected_mask(tokens, sel_mask=sel_mask)
        elif noise == 'no_noise':
            masked_tokens = tokens
        else:
            raise ValueError(f"Noise type ({noise}) not defined.")

        prev_tokens = masked_tokens
        prev_token_mask = prev_tokens.eq(mask_idx)

        return prev_tokens, prev_token_mask

    @staticmethod
    def calculate_accuracy(pred_tokens, target_tokens, mask, chain_token_mask=None):
        # Calculate per-sample accuracy (original functionality)
        per_sample_correct = ((pred_tokens == target_tokens) & mask).sum(dim=1).float()
        per_sample_mask_size = mask.sum(dim=1).float()
        per_sample_accuracy = per_sample_correct / per_sample_mask_size
        
        # Calculate overall accuracy (original functionality)
        correct = (pred_tokens == target_tokens) & mask
        mask_size = mask.sum().float()
        accuracy = correct.sum().float() / mask_size
        
        # Return early if no chain_token_mask is provided
        if chain_token_mask is None:
            return {
                'per_sample_accuracy': per_sample_accuracy, 
                'accuracy': accuracy, 
                'mask_size': mask_size,
            }
        
        # Calculate alpha chain accuracy (chain_token_mask == 1)
        alpha_mask = mask & (chain_token_mask == 1)
        alpha_correct = (pred_tokens == target_tokens) & alpha_mask
        alpha_mask_size = alpha_mask.sum().float()
        alpha_accuracy = alpha_correct.sum().float() / alpha_mask_size if alpha_mask_size > 0 else torch.tensor(0.0, device=mask.device)
        
        # Calculate beta chain accuracy (chain_token_mask == 2)
        beta_mask = mask & (chain_token_mask == 2)
        beta_correct = (pred_tokens == target_tokens) & beta_mask
        beta_mask_size = beta_mask.sum().float()
        beta_accuracy = beta_correct.sum().float() / beta_mask_size if beta_mask_size > 0 else torch.tensor(0.0, device=mask.device)
        
        # Calculate per-sample chain accuracies
        per_sample_alpha_correct = ((pred_tokens == target_tokens) & alpha_mask).sum(dim=1).float()
        per_sample_alpha_mask = alpha_mask.sum(dim=1).float()
        per_sample_alpha_accuracy = torch.zeros_like(per_sample_alpha_correct)
        valid_alpha = per_sample_alpha_mask > 0
        per_sample_alpha_accuracy[valid_alpha] = per_sample_alpha_correct[valid_alpha] / per_sample_alpha_mask[valid_alpha]
        
        per_sample_beta_correct = ((pred_tokens == target_tokens) & beta_mask).sum(dim=1).float()
        per_sample_beta_mask = beta_mask.sum(dim=1).float()
        per_sample_beta_accuracy = torch.zeros_like(per_sample_beta_correct)
        valid_beta = per_sample_beta_mask > 0
        per_sample_beta_accuracy[valid_beta] = per_sample_beta_correct[valid_beta] / per_sample_beta_mask[valid_beta]
        
        return {
            'per_sample_accuracy': per_sample_accuracy, 
            'accuracy': accuracy, 
            'mask_size': mask_size,
            'alpha_accuracy': alpha_accuracy,
            'beta_accuracy': beta_accuracy,
            'alpha_mask_size': alpha_mask_size,
            'beta_mask_size': beta_mask_size,
            'per_sample_alpha_accuracy': per_sample_alpha_accuracy,
            'per_sample_beta_accuracy': per_sample_beta_accuracy
        }
        
    def predict_step(self, batch, noise='random_mask', max_iter=100):
        # inject noise on the input cdr3 tokens
        cdr3_tokens = batch['cdr3_token']
        partial_masks = None
        if noise == 'selected_mask':
            # mask alpha/beta chain independently
            sel_masks, _ = self.mask_partial_chain(batch['chain_token_mask'], mask_prob=1.0)
            batch['cdr3_token'], batch['cdr3_mask'] = self.inject_noise(cdr3_tokens, noise, sel_mask=sel_masks)
            partial_masks = ~sel_masks  # Convert sel_masks to partial_masks (fixed positions)
            # TODO: specify the positions to mask
        else:
            batch['cdr3_token'], batch['cdr3_mask'] = self.inject_noise(cdr3_tokens, noise)
            partial_masks = None

        pred_tokens, pred_scores = self.forward(batch, max_iter, self.sampling_strategy, temperature=self.temperature, use_draft_seq=True, decode_seq=False, partial_masks=partial_masks)
        
        special_sym_mask = (
                cdr3_tokens.eq(self.tokenizer.padding_idx) |
                cdr3_tokens.eq(self.tokenizer.cls_idx) |
                cdr3_tokens.eq(self.tokenizer.eos_idx) |
                cdr3_tokens.eq(self.tokenizer.sep_idx)
            )
        pred_tokens.masked_scatter_(special_sym_mask, cdr3_tokens[special_sym_mask])
        
        # Pass chain_token_mask to calculate_accuracy if it exists in batch
        chain_token_mask = batch.get('chain_token_mask', None)
        acc_results = self.calculate_accuracy(pred_tokens, cdr3_tokens, batch['cdr3_mask'], chain_token_mask)
        
        target_seq = [''.join(seq.split(' ')) for seq in self.tokenizer.batch_decode(cdr3_tokens, remove_special_tokens=True)]
        pred_seq = [''.join(seq.split(' ')) for seq in self.tokenizer.batch_decode(pred_tokens, remove_special_tokens=True)]
        
        for i in range(len(target_seq)):
            print(target_seq[i], pred_seq[i])
        
        # Return the same data as before plus chain-specific metrics if available
        result = {
            'accuracy': acc_results['accuracy'],
            'per_sample_accuracy': acc_results['per_sample_accuracy'],
            'mask_size': acc_results['mask_size'],
            'pred_tokens': pred_tokens,
        }
        
        # Add chain-specific metrics if available
        if chain_token_mask is not None:
            result.update({
                'alpha_accuracy': acc_results['alpha_accuracy'],
                'beta_accuracy': acc_results['beta_accuracy'],
                'alpha_mask_size': acc_results['alpha_mask_size'],
                'beta_mask_size': acc_results['beta_mask_size'],
                'per_sample_alpha_accuracy': acc_results['per_sample_alpha_accuracy'],
                'per_sample_beta_accuracy': acc_results['per_sample_beta_accuracy']
            })
        
        return result
    
    def predict(self, predict_loader, model_location=None, noise='random_mask'):
        avg_acc = 0
        avg_alpha_acc = 0
        avg_beta_acc = 0
        sample_size = 0
        alpha_sample_size = 0
        beta_sample_size = 0
        per_sample_acc_list = []
        per_sample_alpha_acc_list = []
        per_sample_beta_acc_list = []
        
        if model_location is not None:
            self.model.load_state_dict(torch.load(model_location, weights_only=True, map_location='cpu'))
            
        self.model.eval()
        with torch.no_grad():
            for batch in predict_loader:
                for key, value in batch.items():
                    batch[key] = value.to(self.device)
                
                logging_output = self.predict_step(batch, noise, max_iter=self.max_iter)

                num_samples = logging_output['mask_size'].cpu().item()
                avg_acc += logging_output['accuracy'].cpu().item() * num_samples
                per_sample_acc_list.append(logging_output['per_sample_accuracy'])
                sample_size += num_samples
                
                # Add chain-specific metrics if available
                if 'alpha_accuracy' in logging_output:
                    alpha_samples = logging_output['alpha_mask_size'].cpu().item()
                    avg_alpha_acc += logging_output['alpha_accuracy'].cpu().item() * alpha_samples
                    alpha_sample_size += alpha_samples
                    per_sample_alpha_acc_list.append(logging_output['per_sample_alpha_accuracy'])
                    
                    beta_samples = logging_output['beta_mask_size'].cpu().item()
                    avg_beta_acc += logging_output['beta_accuracy'].cpu().item() * beta_samples
                    beta_sample_size += beta_samples
                    per_sample_beta_acc_list.append(logging_output['per_sample_beta_accuracy'])
                    
        result = {
            'accuracy': avg_acc / sample_size,
            'median_accuracy': torch.cat(per_sample_acc_list).median().item(),
            'sample_size': sample_size
        }
        
        # Add chain-specific results if available
        if alpha_sample_size > 0:
            result.update({
                'alpha_accuracy': avg_alpha_acc / alpha_sample_size,
                'median_alpha_accuracy': torch.cat(per_sample_alpha_acc_list).median().item(),
                'alpha_sample_size': alpha_sample_size,
                'beta_accuracy': avg_beta_acc / beta_sample_size,
                'median_beta_accuracy': torch.cat(per_sample_beta_acc_list).median().item(),
                'beta_sample_size': beta_sample_size
            })
        
        return result
    