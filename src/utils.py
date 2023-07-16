import contextlib, time
from typing import NamedTuple
import numpy as np
import torch
import yaml

def read_config(path):
    with open(path, 'r') as f:
        params = yaml.safe_load(f) or {}
    return params

@contextlib.contextmanager
def Timer(name=None, verbose=False):
    start = time.time()
    timer = NamedTuple('timer', elapsed=str)
    yield timer
    timer.elapsed = time.time() - start
    if verbose:
        print(f'{name:<6}: {timer.elapsed:.3f}s')

def str_to_matrix(x):
    x = list(map(float, x.split()))
    return np.array(x).reshape(int(len(x) ** .5), -1)

def strip_outliers(x, alpha=0.025):
    if isinstance(x, np.ndarray):
        l, r = np.quantile(x, q=[alpha, 1 - alpha], axis=0)
        x = np.clip(x, l, r)
    else: # isinstance(torch.Tensor):
        l = torch.quantile(x, q=alpha, axis=0)
        r = torch.quantile(x, q=1 - alpha, axis=0)
        x = torch.clamp(x, l, r)
    return x