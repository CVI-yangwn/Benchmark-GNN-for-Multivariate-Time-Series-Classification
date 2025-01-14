import torch
import numpy as np
from typing import Optional
from sklearn.feature_selection import mutual_info_regression
from functools import lru_cache

adj_matrix = {}
def register_adj(func):
    adj_matrix[func.__name__] = func
    return func

@register_adj
def complete(x: torch.Tensor) -> torch.Tensor:
    # if torch.cuda.is_available():
    #     x = x.cuda()
    num_nodes = x.shape[0]
    return torch.ones(num_nodes, num_nodes)

@register_adj
def identity(x: torch.Tensor) -> torch.Tensor:
    num_nodes = x.shape[0]
    return torch.eye(num_nodes)

@register_adj
def pearson(x: torch.Tensor) -> torch.Tensor:
    if torch.cuda.is_available():
        x = x.cuda()
    return torch.corrcoef(x)

@register_adj
def abspearson(x: torch.Tensor) -> torch.Tensor:
    # if torch.cuda.is_available():
    #     x = x.cuda()
    abspcc = torch.corrcoef(x).abs().nan_to_num()
    # torch.set_printoptions(profile="full")
    # assert not torch.any(torch.isnan(abspcc)), f"has nan. {x}"
    # torch.set_printoptions(profile="default")
    return abspcc

def mi_score(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    xclasses, xclass_idx = torch.unique(x, return_inverse=True)
    yclasses, yclass_idx = torch.unique(y, return_inverse=True)
    n_xclasses = xclasses.shape[0]
    n_yclasses = yclasses.shape[0]

    contingency = torch.sparse_coo_tensor(
        torch.stack([xclass_idx, yclass_idx]),
        torch.ones(xclass_idx.shape[0], device=x.device),
        size=(n_xclasses, n_yclasses)
    )
    contingency = contingency.to_dense()

    nzx, nzy = torch.nonzero(contingency, as_tuple=True)
    nz_val = contingency[nzx, nzy]

    contingency_sum = contingency.sum()
    pi = torch.ravel(contingency.sum(dim=1))
    pj = torch.ravel(contingency.sum(dim=0))

    if pi.size == 1 or pj.size == 1:
        return torch.tensor(0.0)

    log_contingency_nm = torch.log(nz_val)
    contingency_nm = nz_val / contingency_sum
    outer = pi.take(nzx).type(torch.int64) * pj.take(nzy).type(torch.int64)
    log_outer = -torch.log(outer) + torch.log(pi.sum()) + torch.log(pj.sum())
    mi = (
        contingency_nm * (log_contingency_nm - torch.log(contingency_sum))
        + contingency_nm * log_outer
    )
    mi = torch.where(torch.abs(mi) < torch.finfo(mi.dtype).eps, 0.0, mi)
    return torch.clip(mi.sum(), 0.0, None)

import time

# def mutual_information(x: torch.Tensor,
#                       path: Optional[str] = None,
#                       idx: Optional[int] = None) -> torch.Tensor:
@register_adj
def mutual_information(x: torch.Tensor) -> torch.Tensor:
    x = x.numpy()
    v = x.shape[0]
    mi = torch.stack(
            [torch.concat((
                torch.zeros(i),
                torch.stack([torch.tensor(mutual_info_regression(x[i][:, None], x[j]), dtype=torch.float32).squeeze() for j in range(i, v)])))
            for i in range(v)])
    mi = mi + mi.T.multiply(mi.T > mi) - mi.multiply(mi.T > mi)
    # print(mi.shape)
    return mi
# def mutual_information(x: torch.Tensor,
#                       path: Optional[str] = None,
#                       idx: Optional[int] = None) -> torch.Tensor:
#     if path is not None:
#         assert idx is not None, "None index"
#         return torch.load(path)[idx]
#     if torch.cuda.is_available():
#         x = x.cuda()
#     v = x.shape[0]
#     mi = torch.stack(
#             [torch.concat((
#                 torch.zeros(i, device=x.device),
#                 torch.stack([mi_score(x[i], x[j]) for j in range(i, v)])))
#             for i in range(v)])
#     mi = mi + mi.T.multiply(mi.T > mi) - mi.multiply(mi.T > mi)
#     return mi

@register_adj
def phase_locking_value(x: torch.Tensor) -> torch.Tensor:
    from scipy.signal import hilbert
    x = x.numpy()
    v, timepoints = x.shape
    x = np.stack([np.angle(hilbert(np.squeeze(feat))) for feat in x])
    plv = [np.concatenate((
            np.zeros(i),
            np.array([np.abs(np.sum(np.exp(1j*(x[i]-x[j])))) / timepoints
                      for j in range(i, v)]))) for i in range(v)]
    plv = np.stack(plv)
    plv = plv + plv.T*(plv.T > plv) - plv*(plv.T > plv)
    return torch.from_numpy(plv).type(torch.float)

@register_adj
def coherence(x: torch.Tensor) -> torch.Tensor:
    from scipy.signal import coherence
    x = x.numpy()
    coh = np.stack([np.array([np.mean(coherence(ni, nj, 250, "boxcar", 6)[1])
                    for nj in x]) for ni in x])
    return torch.tensor(coh).type(torch.float)


if __name__ == "__main__":
    t1 = time.time()

    x: torch.Tensor = torch.randn(5, 20)
    print(f"dtype of input: {x.dtype}")
    adj = adj_matrix["diffgraphlearn"](x)
    print(f"dtype of output: {adj.dtype}")
    print(adj.shape)

    print("using time:", time.time()-t1)