import torch as th

from stable_baselines3.common.distributions import Distribution as SB3_Distribution


def get_mean_and_chol(p):
    if isinstance(p, th.distributions.Normal):
        return p.mean, p.stddev
    elif isinstance(p, th.distributions.MultivariateNormal):
        return p.mean, p.scale_tril
    elif isinstance(p, SB3_Distribution):
        return get_mean_and_chol(p.distribution)
    else:
        raise Exception('Dist-Type not implemented')


def get_cov(p):
    if isinstance(p, th.distributions.Normal):
        return th.diag(p.variance)
    elif isinstance(p, th.distributions.MultivariateNormal):
        return p.covariance_matrix
    elif isinstance(p, SB3_Distribution):
        return get_cov(p.distribution)
    else:
        raise Exception('Dist-Type not implemented')


def new_dist_like(orig_p, mean, chol):
    if isinstance(orig_p, th.distributions.Normal):
        return th.distributions.Normal(mean, chol)
    elif isinstance(orig_p, th.distributions.MultivariateNormal):
        return th.distributions.MultivariateNormal(mean, scale_tril=chol)
    elif isinstance(orig_p, SB3_Distribution):
        p = orig_p.distribution
        if isinstance(p, th.distributions.Normal):
            p_out = orig_p.__class__(orig_p.action_dim)
            p_out.distribution = th.distributions.Normal(mean, chol)
        elif isinstance(p, th.distributions.MultivariateNormal):
            p_out = orig_p.__class__(orig_p.action_dim)
            p_out.distribution = th.distributions.MultivariateNormal(
                mean, scale_tril=chol)
        else:
            raise Exception('Dist-Type not implemented (of sb3 dist)')
        return p_out
    else:
        raise Exception('Dist-Type not implemented')
