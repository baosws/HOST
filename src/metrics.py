import numpy as np

# https://github.com/vios-s/DiffAN/blob/no_experiments/diffan/utils.py#L7
def OrderDivergence(adj, order):
    err = 0
    for i in range(len(order)):
        err += adj[order[i+1:], order[i]].sum()
    return err

# https://github.com/xunzheng/notears/blob/master/notears/utils.py
def SHD(A_true, A_pred, **kwargs):
    # linear index of nonzeros
    pred = np.flatnonzero(A_pred == 1)
    cond = np.flatnonzero(A_true)
    cond_reversed = np.flatnonzero(A_true.T)
    # reverse
    extra = np.setdiff1d(pred, cond, assume_unique=True)
    reverse = np.intersect1d(extra, cond_reversed, assume_unique=True)
    # structural hamming distance
    pred_lower = np.flatnonzero(np.tril(A_pred + A_pred.T))
    cond_lower = np.flatnonzero(np.tril(A_true + A_true.T))
    extra_lower = np.setdiff1d(pred_lower, cond_lower, assume_unique=True)
    missing_lower = np.setdiff1d(cond_lower, pred_lower, assume_unique=True)
    shd = len(extra_lower) + len(missing_lower) + len(reverse)
    return shd