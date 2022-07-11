from typing import Any, Dict, List, Optional, Tuple, Union
from enum import Enum

import torch as th
from torch import nn
from torch.distributions import Normal, MultivariateNormal

from stable_baselines3.common.preprocessing import get_action_dim

from stable_baselines3.common.distributions import sum_independent_dims
from stable_baselines3.common.distributions import Distribution as SB3_Distribution
from stable_baselines3.common.distributions import DiagGaussianDistribution

from ..misc.fakeModule import FakeModule
from ..misc.distTools import new_dist_like

# TODO: Integrate and Test what I currently have before adding more complexity
# TODO: Support Squashed Dists (tanh)
# TODO: Contextual Cov
# TODO:  - Hybrid
# TODO: Contextual SDE (Scalar + Diag + Full)
# TODO: (SqrtInducedCov (Scalar + Diag + Full))


class Strength(Enum):
    NONE = 0
    SCALAR = 1
    DIAG = 2
    FULL = 3


class ParametrizationType(Enum):
    # Currently only Chol is implemented
    CHOL = 1
    #SPHERICAL_CHOL = 2
    #GIVENS = 3


class EnforcePositiveType(Enum):
    # TODO: Allow custom params for softplus?
    SOFTPLUS = (1, nn.Softplus(beta=1, threshold=20))
    ABS = (2, th.abs)
    RELU = (3, nn.ReLU(inplace=False))
    LOG = (4, th.log)

    def __init__(self, value, func):
        self.value = value
        self._func = func

    def apply(self, x):
        return self._func(x)


class ProbSquashingType(Enum):
    NONE = (0, nn.Identity())
    TANH = (1, th.tanh)

    def __init__(self, value, func):
        self.value = value
        self._func = func

    def apply(self, x):
        return self._func(x)


def get_legal_setups(allowedEPTs=None, allowedParStrength=None, allowedCovStrength=None, allowedPTs=None, allowedPSTs=None):
    allowedEPTs = allowedEPTs or EnforcePositiveType
    allowedParStrength = allowedParStrength or Strength
    allowedCovStrength = allowedCovStrength or Strength
    allowedPTs = allowedPTs or ParametrizationType
    allowedPSTs = allowedPSTs or ProbSquashingType

    for ps in allowedParStrength:
        for cs in allowedCovStrength:
            if ps.value > cs.value:
                continue
            if ps == Strength.SCALAR and cs == Strength.FULL:
                # TODO: Maybe allow?
                continue
            if ps == Strength.NONE:
                yield (ps, cs, None, None)
            else:
                for ept in allowedEPTs:
                    if cs == Strength.FULL:
                        for pt in allowedPTs:
                            yield (ps, cs, ept, pt)
                    else:
                        yield (ps, cs, ept, None)


class UniversalGaussianDistribution(SB3_Distribution):
    """
    Gaussian distribution with configurable covariance matrix shape and optional contextual parametrization mechanism, for continuous actions.

    :param action_dim:  Dimension of the action space.
    """

    def __init__(self, action_dim: int):
        super(UniversalGaussianDistribution, self).__init__()
        self.par_strength = Strength.DIAG
        self.cov_strength = Strength.DIAG
        self.par_type = ParametrizationType.CHOL
        self.enforce_positive_type = EnforcePositiveType.LOG
        self.prob_squashing_type = ProbSquashingType.TANH

        self.distribution = None

    def new_dist_like_me(self, mean, pseudo_chol):
        p = self.distribution
        np = new_dist_like(p, mean, pseudo_chol)
        new = UniversalGaussianDistribution(self.action_dim)
        new.par_strength = self.par_strength
        new.cov_strength = self.cov_strength
        new.par_type = self.par_type
        new.enforce_positive_type = self.enforce_positive_type
        new.prob_squashing_type = self.prob_squashing_type
        new.distribution = np

        return new

    def proba_distribution_net(self, latent_dim: int, log_std_init: float = 0.0) -> Tuple[nn.Module, nn.Module]:
        """
        Create the layers and parameter that represent the distribution:
        one output will be the mean of the Gaussian, the other parameter will be the
        standard deviation (log std in fact to allow negative values)

        :param latent_dim: Dimension of the last layer of the policy (before the action layer)
        :param log_std_init: Initial value for the log standard deviation
        :return: We return two nn.Modules (mean, chol).
        """

        # TODO: Allow chol to be vector when only diagonal.

        mean_actions = nn.Linear(latent_dim, self.action_dim)

        if self.par_strength == Strength.NONE:
            if self.cov_strength == Strength.NONE:
                pseudo_cov_par = th.ones(self.action_dim) * log_std_init
            elif self.cov_strength == Strength.SCALAR:
                pseudo_cov_par = th.ones(self.action_dim) * \
                    nn.Parameter(log_std_init, requires_grad=True)
            elif self.cov_strength == Strength.DIAG:
                pseudo_cov_par = nn.Parameter(
                    th.ones(self.action_dim) * log_std_init, requires_grad=True)
            elif self.cov_strength == Strength.FULL:
                # TODO: This won't work, need to ensure SPD!
                # TODO: Off-axis init?
                pseudo_cov_par = nn.Parameter(
                    th.diag_embed(th.ones(self.action_dim) * log_std_init), requires_grad=True)
            chol = FakeModule(pseudo_cov_par)
        elif self.par_strength == self.cov_strength:
            if self.par_strength == Strength.NONE:
                chol = FakeModule(th.ones(self.action_dim))
            elif self.par_strength == Strength.SCALAR:
                # TODO: Does it work like this? Test!
                std = nn.Linear(latent_dim, 1)
                chol = th.ones(self.action_dim) * std
            elif self.par_strength == Strength.DIAG:
                chol = nn.Linear(latent_dim, self.action_dim)
            elif self.par_strength == Strength.FULL:
                chol = self._parameterize_full(latent_dim)
        elif self.par_strength > self.cov_strength:
            raise Exception(
                'The parameterization can not be stronger than the actual covariance.')
        else:
            if self.par_strength == Strength.SCALAR and self.cov_strength == Strength.DIAG:
                chol = self._parameterize_hybrid_from_scalar(latent_dim)
            elif self.par_strength == Strength.DIAG and self.cov_strength == Strength.FULL:
                chol = self._parameterize_hybrid_from_diag(latent_dim)
            elif self.par_strength == Strength.SCALAR and self.cov_strength == Strength.FULL:
                raise Exception(
                    'That does not even make any sense...')
            else:
                raise Exception("This Exception can't happen (I think)")

        return mean_actions, chol

    def _parameterize_full(self, latent_dim):
        # TODO: Implement various techniques for full parameterization (forcing SPD)
        raise Exception(
            'Programmer-was-to-lazy-to-implement-this-Exception')

    def _parameterize_hybrid_from_diag(self, latent_dim):
        # TODO: Implement the hybrid-method for DIAG -> FULL (parameters for pearson-correlation-matrix)
        raise Exception(
            'Programmer-was-to-lazy-to-implement-this-Exception')

    def _ensure_positive_func(self, x):
        return self.enforce_positive_type.apply(x)

    def _ensure_diagonal_positive(self, pseudo_chol):
        pseudo_chol.tril(-1) + self._ensure_positive_func(pseudo_chol.diagonal(dim1=-2,
                                                                               dim2=-1)).diag_embed() + pseudo_chol.triu(1)

    def _parameterize_hybrid_from_scalar(self, latent_dim):
        factor = nn.Linear(latent_dim, 1)
        par_cov = th.ones(self.action_dim) * \
            nn.Parameter(1, requires_grad=True)
        pseudo_cov = par_cov * factor[0]
        return pseudo_cov

    def proba_distribution(self, mean_actions: th.Tensor, pseudo_cov: th.Tensor) -> "UniversalGaussianDistribution":
        """
        Create the distribution given its parameters (mean, pseudo_cov)

        :param mean_actions:
        :param pseudo_cov:
        :return:
        """
        action_std = None
        # TODO: Needs to be expanded
        if self.cov_strength == Strength.DIAG:
            if self.enforce_positive_type == EnforcePositiveType.LOG:
                action_std = pseudo_cov.exp()
            if action_std == None:
                raise Exception('Not yet implemented!')
            self.distribution = Normal(mean_actions, action_std)
        if self.distribution == None:
            raise Exception('Not yet implemented!')
        return self

    def log_prob(self, actions: th.Tensor) -> th.Tensor:
        """
        Get the log probabilities of actions according to the distribution.
        Note that you must first call the ``proba_distribution()`` method.

        :param actions:
        :return:
        """
        log_prob = self.distribution.log_prob(actions)
        return sum_independent_dims(log_prob)

    def entropy(self) -> th.Tensor:
        return sum_independent_dims(self.distribution.entropy())

    def sample(self) -> th.Tensor:
        # Reparametrization trick to pass gradients
        return self.distribution.rsample()

    def mode(self) -> th.Tensor:
        return self.distribution.mean

    def actions_from_params(self, mean_actions: th.Tensor, log_std: th.Tensor, deterministic: bool = False) -> th.Tensor:
        # Update the proba distribution
        self.proba_distribution(mean_actions, log_std)
        return self.get_actions(deterministic=deterministic)

    def log_prob_from_params(self, mean_actions: th.Tensor, log_std: th.Tensor) -> Tuple[th.Tensor, th.Tensor]:
        """
        Compute the log probability of taking an action
        given the distribution parameters.

        :param mean_actions:
        :param log_std:
        :return:
        """
        actions = self.actions_from_params(mean_actions, log_std)
        log_prob = self.log_prob(actions)
        return actions, log_prob
