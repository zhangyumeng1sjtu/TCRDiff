import random

import torch
import numpy as np
from easydict import EasyDict
import yaml


def set_seed(seed):

    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


def load_config(path):
    with open(path, 'r') as f:
        return EasyDict(yaml.safe_load(f))
