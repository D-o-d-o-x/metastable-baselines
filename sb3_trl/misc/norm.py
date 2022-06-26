import torch as th
from torch.distributions.multivariate_normal import _batch_mahalanobis


def mahalanobis_blub(u, v, std):
    delta = u - v
    return th.triangular_solve(delta, std, upper=False)[0].pow(2).sum([-2, -1])


def mahalanobis(u, v, cov):
    delta = u - v
    return _batch_mahalanobis(cov, delta)
