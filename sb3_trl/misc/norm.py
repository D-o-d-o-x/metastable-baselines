import torch as th
from torch.distributions.multivariate_normal import _batch_mahalanobis


def mahalanobis_blub(u, v, std):
    delta = u - v
    return th.triangular_solve(delta, std, upper=False)[0].pow(2).sum([-2, -1])


def mahalanobis(u, v, cov):
    delta = u - v
    return _batch_mahalanobis(cov, delta)


def frob_sq(diff, is_spd=False):
    # If diff is spd, we can use a more perfromant algorithm
    if is_spd:
        return _frob_sq_spd(diff)
    return th.norm(diff, p='fro', dim=tuple(range(1, diff.dim()))).pow(2)


def _frob_sq_spd(diff):
    return _batch_trace(diff @ diff)


def _batch_trace(x):
    return th.diagonal(x, dim1=-2, dim2=-1).sum(-1)
