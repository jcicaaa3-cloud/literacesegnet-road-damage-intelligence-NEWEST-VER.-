import os
import random

import numpy as np
import torch
import yaml


def load_yaml(path):
  with open(path, 'r', encoding='utf-8') as f:
    return yaml.safe_load(f)


def make_dir(path):
  os.makedirs(path, exist_ok=True)


def set_seed(seed):
  random.seed(seed)
  np.random.seed(seed)
  torch.manual_seed(seed)
  torch.cuda.manual_seed_all(seed)
  torch.backends.cudnn.deterministic = False
  torch.backends.cudnn.benchmark = True


def get_device(cfg):
  want = cfg.get('device', 'cuda')
  if want == 'cuda' and torch.cuda.is_available():
    return torch.device('cuda')
  return torch.device('cpu')


def count_params(model):
  return sum(p.numel() for p in model.parameters() if p.requires_grad)


def show_score(score):
  parts = []
  for k, v in score.items():
    if isinstance(v, float):
      parts.append(f'{k}={v:.4f}')
    else:
      parts.append(f'{k}={v}')
  return ' | '.join(parts)
