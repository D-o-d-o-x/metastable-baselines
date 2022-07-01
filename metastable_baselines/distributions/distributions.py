from typing import Any, Dict, List, Optional, Tuple, Union
from enum import Enum

import torch as th
from torch import nn
from torch.distributions import Normal, MultivariateNormal

from stable_baselines3.common.preprocessing import get_action_dim

from stable_baselines3.common.distributions import sum_independent_dims
from stable_baselines3.common.distributions import Distribution as SB3_Distribution
from stable_baselines3.common.distributions import DiagGaussianDistribution


# TODO: Full Cov Parameter
# TODO: Contextual Cov
# TODO:  - Scalar
# TODO:  - Diag
# TODO:  - Full
# TODO:  - Hybrid
# TODO: Contextual SDE (Scalar + Diag + Full)
# TODO: (SqrtInducedCov (Scalar + Diag + Full))
# TODO: (Support Squased Dists (tanh))

class Strength(Enum):
    NONE = 0
    SCALAR = 1
    DIAG = 2
    FULL = 3

    def __init__(self, num):
        self.num = num

    @property
    def foo(self):
        return self.num


class ParametrizationType(Enum):
    CHOL = 1
    ARCHAKOVA = 2


class EnforcePositiveType(Enum):
    LOG = 1
    RELU = 2
    SELU = 3
    ABS = 4
    SQ = 5


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

        self.distribution = None

    def proba_distribution_net(self, latent_dim: int, log_std_init: float = 0.0) -> Tuple[nn.Module, nn.Parameter]:
        """
        Create the layers and parameter that represent the distribution:
        one output will be the mean of the Gaussian, the other parameter will be the
        standard deviation (log std in fact to allow negative values)

        :param latent_dim: Dimension of the last layer of the policy (before the action layer)
        :param log_std_init: Initial value for the log standard deviation
        :return:
        """
        mean_actions = nn.Linear(latent_dim, self.action_dim)

        if self.par_strength == Strength.NONE:
            if self.cov_strength == Strength.NONE:
                pseudo_cov = th.ones(self.action_dim) * log_std_init
            elif self.cov_strength == Strength.SCALAR:
                pseudo_cov = th.ones(self.action_dim) * \
                    nn.Parameter(log_std_init, requires_grad=True)
            elif self.cov_strength == Strength.DIAG:
                pseudo_cov = nn.Parameter(
                    th.ones(self.action_dim) * log_std_init, requires_grad=True)
            elif self.cov_strength == Strength.FULL:
                # Off-axis init?
                pseudo_cov = nn.Parameter(
                    th.diag_embed(th.ones(self.action_dim) * log_std_init), requires_grad=True)
        elif self.par_strength == self.cov_strength:
            if self.par_strength == Strength.NONE:
                pseudo_cov = th.ones(self.action_dim)
            elif self.par_strength == Strength.SCALAR:
                std = nn.Linear(latent_dim, 1)
                pseudo_cov = th.ones(self.action_dim) * std
            elif self.par_strength == Strength.DIAG:
                pseudo_cov = nn.Linear(latent_dim, self.action_dim)
            elif self.par_strength == Strength.FULL:
                raise Exception("Don't know how to implement yet...")
        elif self.par_strength > self.cov_strength:
            raise Exception(
                'The parameterization can not be stronger than the actual covariance.')
        else:
            if self.par_strength == Strength.SCALAR and self.cov_strength == Strength.DIAG:
                factor = nn.Linear(latent_dim, 1)
                par_cov = th.ones(self.action_dim) * \
                    nn.Parameter(1, requires_grad=True)
                pseudo_cov = par_cov * factor[0]
            elif self.par_strength == Strength.SCALAR and self.cov_strength == Strength.FULL:
                raise Exception(
                    'That does not even make any sense...')
            else:
                raise Exception(
                    'Programmer-was-to-lazy-to-implement-this-Exception')

        return mean_actions, pseudo_cov

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
