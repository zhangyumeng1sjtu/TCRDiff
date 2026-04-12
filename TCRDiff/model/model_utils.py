import os
import math

import torch
import torch.nn.functional as F
import einx
from einops import pack, unpack


# helper functions
def exists(v):
    return v is not None


def default(v, d):
    return v if exists(v) else d


def softclamp(t, value):
    return (t / value).tanh() * value


def max_neg_value(t):
    return -torch.finfo(t.dtype).max


def to_pairwise_mask(
    mask_i,  # (..., n)
    mask_j=None  # (..., n) | None -> (..., n, n)
):
    mask_j = default(mask_j, mask_i)
    assert mask_i.shape == mask_j.shape
    return einx.logical_and('... i, ... j -> ... i j', mask_i, mask_j)


def pad_at_dim(
    t,
    pad,
    *,
    dim=-1,
    value=0.
):
    dims_from_right = (- dim - 1) if dim < 0 else (t.ndim - dim - 1)
    zeros = ((0, 0) * dims_from_right)
    return F.pad(t, (*zeros, *pad), value=value)


def pack_one(t, pattern):
    """Pack a single tensor into a tuple of tensors with the given pattern.

    :param t: The tensor to pack.
    :param pattern: The pattern with which to pack.
    :return: The packed tensor along with the shape(s) of the tensor.
    """
    packed, ps = pack([t], pattern)

    def unpack_one(to_unpack, unpack_pattern=None):
        """Unpack a single tensor.

        :param to_unpack: The tensor to unpack.
        :param pattern: The pattern with which to unpack.
        :return: The unpacked tensor.
        """
        (unpacked,) = unpack(to_unpack, ps, default(unpack_pattern, pattern))
        return unpacked

    return packed, unpack_one


def get_index_embedding(indices, embed_size, max_len=128):
    """Creates sine / cosine positional embeddings from a prespecified indices.

    Args:
        indices: offsets of size [..., N_edges] of type integer
        max_len: maximum length.
        embed_size: dimension of the embeddings to create

    Returns:
        positional embedding of shape [N, embed_size]
    """
    K = torch.arange(embed_size//2).to(indices.device)
    pos_embedding_sin = torch.sin(
        indices[..., None] * math.pi / (max_len**(2*K[None]/embed_size))).to(indices.device)
    pos_embedding_cos = torch.cos(
        indices[..., None] * math.pi / (max_len**(2*K[None]/embed_size))).to(indices.device)
    pos_embedding = torch.cat([pos_embedding_sin, pos_embedding_cos], axis=-1)
    return pos_embedding


class EarlyStopping:

    def __init__(self, patience=10, checkpoint_dir='logs'):
        self.patience = patience
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.checkpoint_dir = checkpoint_dir

    def __call__(self, score, model, goal="maximize"):

        if goal == "minimize":
            score = -score

        if self.best_score is None:
            self.best_score = score
            self.save_checkpoint(score, model)

        elif score < self.best_score:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            self.save_checkpoint(score, model)
            self.counter = 0

    def save_checkpoint(self, score, model):
        torch.save(model.state_dict(), os.path.join(
            self.checkpoint_dir, 'checkpoint.pt'))
        self.best_score = score


def sample_from_categorical(logits=None, temperature=1.0): # small temperature -> less randomness
    if temperature:
        dist = torch.distributions.Categorical(logits=logits.div(temperature))
        tokens = dist.sample()

        # scores = dist.log_prob(tokens)

        # compute scores based on original logits
        log_probs = F.log_softmax(logits, dim=-1)
        scores = log_probs.gather(dim=-1, index=tokens.unsqueeze(-1)).squeeze(-1)

    else:
        scores, tokens = logits.log_softmax(dim=-1).max(dim=-1)

    return tokens, scores
    

def stochastic_sample_from_categorical(logits=None, temperature=1.0, noise_scale=1.0):
    gumbel_noise = -torch.log(-torch.log(torch.rand_like(logits) + 1e-8) + 1e-8)
    logits = logits + noise_scale * gumbel_noise

    tokens, scores = sample_from_categorical(logits, temperature)
    # scores, tokens = logits.log_softmax(dim=-1).max(dim=-1)
    return tokens, scores


def topk_masking(scores, cutoff_len, stochastic=False, temp=1.0):
    """
    scores: [b, n]
    cutoff_len: [b, 1]
    stochastic: bool, whether to add noise to select top_k or not
    returns:
        mask: [b, n], with 1 if the token is in top-k lowest scores, 0 otherwise
    """
    if stochastic:
        gumbel_noise = -torch.log(-torch.log(torch.rand_like(scores) + 1e-8) + 1e-8)
        _scores = scores + temp * gumbel_noise
    else:
        _scores = scores
    sorted_index = _scores.sort(-1)[0]
    cutoff = sorted_index.gather(dim=-1, index=cutoff_len)
    masking = _scores < cutoff
    return masking


def top_k_top_p_filtering(logits, top_k=0, top_p=0.95, filter_value=-float('Inf')):
    """ Filter a distribution of logits using top-k and/or nucleus (top-p) filtering
        Args:
            logits: logits distribution shape (vocabulary size)
            top_k >0: keep only top k tokens with highest probability (top-k filtering).
            top_p >0.0: keep the top tokens with cumulative probability >= top_p (nucleus filtering).
                Nucleus filtering is described in Holtzman et al. (http://arxiv.org/abs/1904.09751)
        Basic outline taken from https://gist.github.com/thomwolf/1a5a29f6962089e871b94cbd09daf317
    """
    ori_shape = logits.shape
    logits = logits.reshape(-1, ori_shape[-1])
    assert logits.dim() == 2  # [BATCH_SIZE, VOCAB_SIZE]
    top_k = min(top_k, logits.size(-1))  # Safety check
    if top_k > 0:
        # Remove all tokens with a probability less than the last token of the top-k
        indices_to_remove = logits < torch.topk(logits, top_k, dim=1)[0][..., -1, None]
        logits[indices_to_remove] = filter_value
    sorted_logits, sorted_indices = torch.sort(logits, descending=True)
    cumulative_probs = torch.cumsum(torch.softmax(sorted_logits, dim=-1), dim=-1)
    # Remove tokens with cumulative probability above the threshold
    sorted_indices_to_remove = cumulative_probs > top_p
    # Shift the indices to the right to keep also the first token above the threshold
    sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
    sorted_indices_to_remove[..., 0] = 0
    # Replace logits to be removed with -inf in the sorted_logits
    sorted_logits[sorted_indices_to_remove] = filter_value
    # Then reverse the sorting process by mapping back sorted_logits to their original position
    logits = torch.gather(sorted_logits, 1, sorted_indices.argsort(-1))
    logits = logits.reshape(ori_shape) 
    return logits
