import torch


def recursive_apply(x, fn):
    if isinstance(x, dict):
        return {k: recursive_apply(v, fn) for k, v in x.items()}
    elif isinstance(x, (tuple, list)):
        return x.__class__([recursive_apply(e, fn) for e in x])
    else:
        assert isinstance(x, torch.Tensor)
        return fn(x)
