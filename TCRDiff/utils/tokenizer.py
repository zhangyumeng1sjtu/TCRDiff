import itertools
from typing import Sequence, List
from abc import abstractmethod

import torch


PROTEIN_SEQ_TOKS = ['A', 'C', 'D', 'E', 'F', 'G', 'H', 'I',
                    'K', 'L', 'M', 'N', 'P', 'Q', 'R', 'S', 'T', 'V', 'W', 'Y']


class BaseTokenizer(object):

    def __init__(
        self,
        standard_toks: Sequence[str] = PROTEIN_SEQ_TOKS,
        prepend_toks: Sequence[str] = ("<pad>", "<eos>", "<unk>"),
        append_toks: Sequence[str] = ("<cls>", "X", "<sep>"),
        prepend_bos: bool = True,
        append_eos: bool = False
    ):
        self.standard_toks = list(standard_toks)
        self.prepend_toks = list(prepend_toks)
        self.append_toks = list(append_toks)
        self.prepend_bos = prepend_bos
        self.append_eos = append_eos

        self.all_toks = self.prepend_toks + self.standard_toks + self.append_toks
        self.tok_to_idx = {tok: i for i, tok in enumerate(self.all_toks)}

        self.unk_idx = self.tok_to_idx["<unk>"]
        self.padding_idx = self.get_idx("<pad>")
        self.sep_idx = self.get_idx("<sep>")
        self.cls_idx = self.get_idx("<cls>")
        self.mask_idx = self.get_idx("X")
        self.eos_idx = self.get_idx("<eos>")
        
        self.all_special_toks = ['<eos>', '<unk>', '<pad>', '<sep>', '<cls>', 'X']
        self.speical_token_idxes = [self.tok_to_idx[tok] for tok in self.all_special_toks]
        self.standard_token_idxes = [self.tok_to_idx[tok] for tok in self.standard_toks]
        self.unique_no_split_tokens = self.all_toks

    def __len__(self):
        return len(self.all_toks)

    def get_idx(self, tok):
        return self.tok_to_idx.get(tok, self.unk_idx)

    def get_tok(self, idx):
        return self.all_toks[idx]

    def to_dict(self):
        return self.tok_to_idx.copy()

    def _tokenize(self, text) -> str:
        return text.split()

    def tokenize(self, text, **kwargs) -> List[str]:

        def split_on_token(tok, text):
            result = []
            split_text = text.split(tok)
            for i, sub_text in enumerate(split_text):
                if i < len(split_text) - 1:
                    sub_text = sub_text.rstrip()
                if i > 0:
                    sub_text = sub_text.lstrip()

                if i == 0 and not sub_text:
                    result.append(tok)
                elif i == len(split_text) - 1:
                    if sub_text:
                        result.append(sub_text)
                    else:
                        pass
                else:
                    if sub_text:
                        result.append(sub_text)
                    result.append(tok)
            return result

        def split_on_tokens(tok_list, text):
            if not text.strip():
                return []

            tokenized_text = []
            text_list = [text]
            for tok in tok_list:
                tokenized_text = []
                for sub_text in text_list:
                    if sub_text not in self.unique_no_split_tokens:
                        tokenized_text.extend(split_on_token(tok, sub_text))
                    else:
                        tokenized_text.append(sub_text)
                text_list = tokenized_text

            return list(
                itertools.chain.from_iterable(
                    (
                        self._tokenize(token)
                        if token not in self.unique_no_split_tokens
                        else [token]
                        for token in tokenized_text
                    )
                )
            )

        no_split_token = self.unique_no_split_tokens
        tokenized_text = split_on_tokens(no_split_token, text)
        return tokenized_text

    @abstractmethod
    def encode(self, text):
        raise NotImplementedError

    @abstractmethod
    def mask_token(self, tokens, mask_prob, **kwargs):
        raise NotImplementedError
    
    
class PeptideTokenizer(BaseTokenizer):
    def __init__(self, **kwargs):
        super(PeptideTokenizer, self).__init__(**kwargs)
        
    def encode(self, text):
        return [self.tok_to_idx[tok] for tok in self.tokenize(text)]
    
    def mask_token(self, tokens, mask_prob=0.15):
        labels = tokens.clone()
        prob_matrix = torch.full(labels.shape, mask_prob)

        special_token_mask = torch.isin(labels, torch.tensor(self.speical_token_idxes))
        prob_matrix.masked_fill_(special_token_mask, value=0.0)

        masked_idxes = torch.bernoulli(prob_matrix).bool()
        tokens[masked_idxes] = self.mask_idx
        
        # Ensure each sequence has at least one masked token (excluding special tokens)
        for i in range(tokens.shape[0]):  # assuming shape [batch_size, seq_length]
            if not masked_idxes[i].any():  # if no token is masked in this sequence
                # Get indices that are not special tokens
                non_special_indices = (~special_token_mask[i]).nonzero(as_tuple=True)[0]
                if len(non_special_indices) > 0:  # Ensure there is at least one non-special token
                    random_idx = non_special_indices[torch.randint(len(non_special_indices), (1,))]
                    masked_idxes[i, random_idx] = True  # mask this token
                    tokens[i, random_idx] = self.mask_idx  # apply the mask
                
        return tokens, labels
    
    def mask_token_(self, tokens, mask_prob=0.15, no_mask_ratio=0.5):
        labels = tokens.clone()
        prob_matrix = torch.full(labels.shape, mask_prob)
        
        # Randomly decide rows to be completely unmasked
        no_mask_rows = torch.rand(labels.size(0)) < no_mask_ratio
        prob_matrix[no_mask_rows] = 0  # Set mask probability to 0 for these rows
        
        masked_idxes = torch.bernoulli(prob_matrix).bool()
        tokens[masked_idxes] = self.mask_idx
        
        return tokens, labels
    
    
class PairCDR3Tokenizer(BaseTokenizer):
    def __init__(self, **kwargs):
        super(PairCDR3Tokenizer, self).__init__(**kwargs)

        # self.all_toks = self.prepend_toks + self.standard_toks + self.append_toks + ['<na>']
        # self.tok_to_idx = {tok: i for i, tok in enumerate(self.all_toks)}

        # self.unk_idx = self.tok_to_idx["<unk>"]
        # self.padding_idx = self.get_idx("<pad>")
        # self.sep_idx = self.get_idx("<sep>")
        # self.cls_idx = self.get_idx("<cls>")
        # self.mask_idx = self.get_idx("X")
        # self.eos_idx = self.get_idx("<eos>")
        # self.na_chain_idx = self.get_idx("<na>")
        
        # self.all_special_toks = ['<eos>', '<unk>', '<pad>', '<sep>', '<cls>', 'X', '<na>']
        # self.speical_token_idxes = [self.tok_to_idx[tok] for tok in self.all_special_toks]
        # self.standard_token_idxes = [self.tok_to_idx[tok] for tok in self.standard_toks]
        # self.unique_no_split_tokens = self.all_toks
        
    def encode(self, alpha_seq, beta_seq):
        alpha_tokens = [self.tok_to_idx[tok] for tok in self.tokenize(alpha_seq)]
        beta_tokens = [self.tok_to_idx[tok] for tok in self.tokenize(beta_seq)]
        return alpha_tokens + [self.sep_idx] + beta_tokens
    
    def decode(self, token_ids: List[int], remove_special_tokens: bool = True) -> str:
        tokens = ''.join([self.get_tok(idx) for idx in token_ids])
        if remove_special_tokens:
            tokens = tokens.replace(self.get_tok(self.mask_idx), '_') \
                .replace(self.get_tok(self.eos_idx), '') \
                .replace(self.get_tok(self.cls_idx), '') \
                .replace(self.get_tok(self.padding_idx), '') \
                .replace(self.get_tok(self.unk_idx), '-') \
                .replace(self.get_tok(self.sep_idx), '|')
        return " ".join(tokens)
    
    def batch_decode(
        self, 
        batch_token_ids: List[List[int]], 
        remove_special_tokens: bool = True
    ) -> List[str]:
        decoded_sequences = []
        for token_ids in batch_token_ids:
            decoded_str = self.decode(token_ids, remove_special_tokens=remove_special_tokens)
            decoded_sequences.append(decoded_str)
        return decoded_sequences
    
    def mask_token(self, tokens, mask_prob=0.15):
        labels = tokens.clone()
        prob_matrix = torch.full(labels.shape, mask_prob)

        special_token_mask = torch.isin(labels, torch.tensor(self.speical_token_idxes))
        prob_matrix.masked_fill_(special_token_mask, value=0.0)

        masked_idxes = torch.bernoulli(prob_matrix).bool()
        tokens[masked_idxes] = self.mask_idx
        
        # Ensure each sequence has at least one masked token (excluding special tokens)
        for i in range(tokens.shape[0]):  # assuming shape [batch_size, seq_length]
            if not masked_idxes[i].any():  # if no token is masked in this sequence
                # Get indices that are not special tokens
                non_special_indices = (~special_token_mask[i]).nonzero(as_tuple=True)[0]
                if len(non_special_indices) > 0:  # Ensure there is at least one non-special token
                    random_idx = non_special_indices[torch.randint(len(non_special_indices), (1,))]
                    masked_idxes[i, random_idx] = True  # mask this token
                    tokens[i, random_idx] = self.mask_idx  # apply the mask
                
        return tokens, labels
    
    def mask_token_(self, tokens, chain_token_mask, mask_prob=0.15, mask_chain_prob=0.15):
        labels = tokens.clone()
        prob_matrix = torch.full(labels.shape, mask_prob)

        special_token_mask = torch.isin(labels, torch.tensor(self.speical_token_idxes))
        prob_matrix.masked_fill_(special_token_mask, value=0.0)
        
        # Randomly select samples to apply whole-chain masking
        num_samples = tokens.size(0)
        masked_chain_indices = torch.zeros(num_samples, dtype=torch.long)
        masked_samples = torch.rand(num_samples) < mask_chain_prob
        
        # Randomly choose which chain (alpha or beta) to mask for selected samples
        masked_chain = torch.randint(0, 2, (num_samples,))
        alpha_chain_mask = (chain_token_mask == 1)
        beta_chain_mask = (chain_token_mask == 2)
        
        masked_chain_indices[masked_samples & (masked_chain == 0)] = 1  # Mark alpha chain for masking
        masked_chain_indices[masked_samples & (masked_chain == 1)] = 2  # Mark beta chain for masking 
        
        # Set mask probability to 1.0 for the chosen chain in selected samples
        prob_matrix[masked_chain_indices == 1, :] = alpha_chain_mask[masked_chain_indices == 1].float()
        prob_matrix[masked_chain_indices == 2, :] = beta_chain_mask[masked_chain_indices == 2].float()
        
        masked_idxes = torch.bernoulli(prob_matrix).bool()
        tokens[masked_idxes] = self.mask_idx
        
        # Ensure each sequence has at least one masked token (excluding special tokens)
        for i in range(tokens.shape[0]):  # assuming shape [batch_size, seq_length]
            if not masked_idxes[i].any():  # if no token is masked in this sequence
                # Get indices that are not special tokens
                non_special_indices = (~special_token_mask[i]).nonzero(as_tuple=True)[0]
                if len(non_special_indices) > 0:  # Ensure there is at least one non-special token
                    random_idx = non_special_indices[torch.randint(len(non_special_indices), (1,))]
                    masked_idxes[i, random_idx] = True  # mask this token
                    tokens[i, random_idx] = self.mask_idx  # apply the mask
                
        return tokens, labels, masked_chain_indices
