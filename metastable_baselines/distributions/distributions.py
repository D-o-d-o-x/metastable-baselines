from typing import Any, Dict, List, Optional, Tuple, Union
from enum import Enum

import gym
import torch as th
from torch import nn
from torch.distributions import Normal, MultivariateNormal
from math import pi

from stable_baselines3.common.preprocessing import get_action_dim

from stable_baselines3.common.distributions import sum_independent_dims
from stable_baselines3.common.distributions import Distribution as SB3_Distribution
from stable_baselines3.common.distributions import (
    BernoulliDistribution,
    CategoricalDistribution,
    MultiCategoricalDistribution,
    #    StateDependentNoiseDistribution,
)
from stable_baselines3.common.distributions import DiagGaussianDistribution

from ..misc.tensor_ops import fill_triangular

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
    CHOL = 1
    SPHERICAL_CHOL = 2
    # Not (yet?) implemented:
    #GIVENS = 3
    #NNLN_EIGEN = 4


class EnforcePositiveType(Enum):
    # TODO: Allow custom params for softplus?
    SOFTPLUS = (1, nn.Softplus(beta=1, threshold=20))
    ABS = (2, th.abs)
    RELU = (3, nn.ReLU(inplace=False))
    LOG = (4, th.log)

    def __init__(self, value, func):
        self.val = value
        self._func = func

    def apply(self, x):
        return self._func(x)


class ProbSquashingType(Enum):
    NONE = (0, nn.Identity())
    TANH = (1, th.tanh)

    def __init__(self, value, func):
        self.val = value
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
            if ps == Strength.DIAG and cs == Strength.FULL:
                # TODO: Implement
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


def make_proba_distribution(
    action_space: gym.spaces.Space, use_sde: bool = False, dist_kwargs: Optional[Dict[str, Any]] = None
) -> SB3_Distribution:
    """
    Return an instance of Distribution for the correct type of action space
    :param action_space: the input action space
    :param use_sde: Force the use of StateDependentNoiseDistribution
        instead of DiagGaussianDistribution
    :param dist_kwargs: Keyword arguments to pass to the probability distribution
    :return: the appropriate Distribution object
    """
    if dist_kwargs is None:
        dist_kwargs = {}

    if isinstance(action_space, gym.spaces.Box):
        assert len(
            action_space.shape) == 1, "Error: the action space must be a vector"
        return UniversalGaussianDistribution(get_action_dim(action_space), use_sde=use_sde, **dist_kwargs)
    elif isinstance(action_space, gym.spaces.Discrete):
        return CategoricalDistribution(action_space.n, **dist_kwargs)
    elif isinstance(action_space, gym.spaces.MultiDiscrete):
        return MultiCategoricalDistribution(action_space.nvec, **dist_kwargs)
    elif isinstance(action_space, gym.spaces.MultiBinary):
        return BernoulliDistribution(action_space.n, **dist_kwargs)
    else:
        raise NotImplementedError(
            "Error: probability distribution, not implemented for action space"
            f"of type {type(action_space)}."
            " Must be of type Gym Spaces: Box, Discrete, MultiDiscrete or MultiBinary."
        )


class UniversalGaussianDistribution(SB3_Distribution):
    """
    Gaussian distribution with configurable covariance matrix shape and optional contextual parametrization mechanism, for continuous actions.

    :param action_dim:  Dimension of the action space.
    """

    def __init__(self, action_dim: int, use_sde: bool = False, neural_strength: Strength = Strength.DIAG, cov_strength: Strength = Strength.DIAG, parameterization_type: ParametrizationType = ParametrizationType.CHOL, enforce_positive_type: EnforcePositiveType = EnforcePositiveType.ABS, prob_squashing_type: ProbSquashingType = ProbSquashingType.NONE):
        super(UniversalGaussianDistribution, self).__init__()
        self.action_dim = action_dim
        self.par_strength = neural_strength
        self.cov_strength = cov_strength
        self.par_type = parameterization_type
        self.enforce_positive_type = enforce_positive_type
        self.prob_squashing_type = prob_squashing_type

        self.distribution = None

        if self.prob_squashing_type != ProbSquashingType.NONE:
            raise Exception('ProbSquasing is not yet implmenented!')

        if use_sde:
            raise Exception('SDE is not yet implemented')

    def new_dist_like_me(self, mean: th.Tensor, chol: th.Tensor):
        p = self.distribution
        if isinstance(p, th.distributions.Normal):
            if p.stddev.shape != chol.shape:
                chol = th.diagonal(chol, dim1=1, dim2=2)
            np = th.distributions.Normal(mean, chol)
        elif isinstance(p, th.distributions.MultivariateNormal):
            np = th.distributions.MultivariateNormal(mean, scale_tril=chol)
        new = UniversalGaussianDistribution(self.action_dim, neural_strength=self.par_strength, cov_strength=self.cov_strength,
                                            parameterization_type=self.par_strength, enforce_positive_type=self.enforce_positive_type, prob_squashing_type=self.prob_squashing_type)
        new.distribution = np

        return new

    def proba_distribution_net(self, latent_dim: int, latent_sde_dim: int, std_init: float = 0.0) -> Tuple[nn.Module, nn.Module]:
        """
        Create the layers and parameter that represent the distribution:
        one output will be the mean of the Gaussian, the other parameter will be the
        standard deviation

        :param latent_dim: Dimension of the last layer of the policy (before the action layer)
        :param std_init: Initial value for the standard deviation
        :return: We return two nn.Modules (mean, chol). chol can be a vector if the full chol would be a diagonal.
        """

        assert std_init >= 0.0, "std can not be initialized to a negative value."

        # TODO: Implement SDE
        self.latent_sde_dim = latent_sde_dim

        mean_actions = nn.Linear(latent_dim, self.action_dim)
        chol = CholNet(latent_dim, self.action_dim, std_init, self.par_strength,
                       self.cov_strength, self.par_type, self.enforce_positive_type, self.prob_squashing_type)

        return mean_actions, chol

    def proba_distribution(self, mean_actions: th.Tensor, chol: th.Tensor, latent_pi: nn.Module) -> "UniversalGaussianDistribution":
        """
        Create the distribution given its parameters (mean, chol)

        :param mean_actions:
        :param chol:
        :return:
        """
        # TODO: latent_pi is for SDE, implement.

        if self.cov_strength in [Strength.NONE, Strength.SCALAR, Strength.DIAG]:
            self.distribution = Normal(mean_actions, chol)
        elif self.cov_strength in [Strength.FULL]:
            self.distribution = MultivariateNormal(mean_actions, cholesky=chol)
        if self.distribution == None:
            raise Exception('Unable to create torch distribution')
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


class CholNet(nn.Module):
    def __init__(self, latent_dim: int, action_dim: int, std_init: float, par_strength: Strength, cov_strength: Strength, par_type: ParametrizationType, enforce_positive_type: EnforcePositiveType, prob_squashing_type: ProbSquashingType):
        super().__init__()
        self.latent_dim = latent_dim
        self.action_dim = action_dim

        self.par_strength = par_strength
        self.cov_strength = cov_strength
        self.par_type = par_type
        self.enforce_positive_type = enforce_positive_type
        self.prob_squashing_type = prob_squashing_type

        self._flat_chol_len = action_dim * (action_dim + 1) // 2

        # Yes, this is ugly.
        # But I don't know how this mess could be elegantly abstracted away...

        if self.par_strength == Strength.NONE:
            if self.cov_strength == Strength.NONE:
                self.chol = th.ones(self.action_dim) * std_init
            elif self.cov_strength == Strength.SCALAR:
                self.param = nn.Parameter(std_init, requires_grad=True)
            elif self.cov_strength == Strength.DIAG:
                self.params = nn.Parameter(
                    th.ones(self.action_dim) * std_init, requires_grad=True)
            elif self.cov_strength == Strength.FULL:
                # TODO: Init Off-axis differently?
                self.params = nn.Parameter(
                    th.ones(self._full_params_len) * std_init, requires_grad=True)
        elif self.par_strength == self.cov_strength:
            if self.par_strength == Strength.SCALAR:
                self.std = nn.Linear(latent_dim, 1)
            elif self.par_strength == Strength.DIAG:
                self.diag_chol = nn.Linear(latent_dim, self.action_dim)
            elif self.par_strength == Strength.FULL:
                self.params = nn.Linear(latent_dim, self._full_params_len)
        elif self.par_strength > self.cov_strength:
            raise Exception(
                'The parameterization can not be stronger than the actual covariance.')
        else:
            if self.par_strength == Strength.SCALAR and self.cov_strength == Strength.DIAG:
                self.factor = nn.Linear(latent_dim, 1)
                self.param = nn.Parameter(1, requires_grad=True)
            elif self.par_strength == Strength.DIAG and self.cov_strength == Strength.FULL:
                # TODO
                pass
            elif self.par_strength == Strength.SCALAR and self.cov_strength == Strength.FULL:
                # TODO
                pass
            else:
                raise Exception("This Exception can't happen")

    def forward(self, x: th.Tensor) -> th.Tensor:
        # Ugly mess pt.2:
        if self.par_strength == Strength.NONE:
            if self.cov_strength == Strength.NONE:
                return self.chol
            elif self.cov_strength == Strength.SCALAR:
                return self._ensure_positive_func(
                    th.ones(self.action_dim) * self.param)
            elif self.cov_strength == Strength.DIAG:
                return self._ensure_positive_func(self.params)
            elif self.cov_strength == Strength.FULL:
                return self._parameterize_full(self.params)
        elif self.par_strength == self.cov_strength:
            if self.par_strength == Strength.SCALAR:
                std = self.std(x)
                diag_chol = th.ones(self.action_dim) * std
                return self._ensure_positive_func(diag_chol)
            elif self.par_strength == Strength.DIAG:
                diag_chol = self.diag_chol(x)
                return self._ensure_positive_func(diag_chol)
            elif self.par_strength == Strength.FULL:
                params = self.params(x)
                return self._parameterize_full(params)
        else:
            if self.par_strength == Strength.SCALAR and self.cov_strength == Strength.DIAG:
                factor = self.factor(x)
                diag_chol = self._ensure_positive_func(
                    th.ones(self.action_dim) * self.param * factor[0])
                return diag_chol
            elif self.par_strength == Strength.DIAG and self.cov_strength == Strength.FULL:
                pass
                # TODO
            elif self.par_strength == Strength.SCALAR and self.cov_strength == Strength.FULL:
                # TODO
                pass
        raise Exception()

    @property
    def _full_params_len(self):
        if self.par_type == ParametrizationType.CHOL:
            return self._flat_chol_len
        elif self.par_type == ParametrizationType.SPHERICAL_CHOL:
            return self._flat_chol_len
        raise Exception()

    def _parameterize_full(self, params):
        if self.par_type == ParametrizationType.CHOL:
            return self._chol_from_flat(params)
        elif self.par_type == ParametrizationType.SPHERICAL_CHOL:
            return self._chol_from_flat_sphe_chol(params)
        raise Exception()

    def _chol_from_flat(self, flat_chol):
        chol = fill_triangular(flat_chol).expand(self._flat_chol_len, -1, -1)
        return self._ensure_diagonal_positive(chol)

    def _chol_from_flat_sphe_chol(self, flat_sphe_chol):
        pos_flat_sphe_chol = self._ensure_positive_func(flat_sphe_chol)
        sphe_chol = fill_triangular(pos_flat_sphe_chol).expand(
            self._flat_chol_len, -1, -1)
        chol = self._chol_from_sphe_chol(sphe_chol)
        return chol

    def _chol_from_sphe_chol(self, sphe_chol):
        # TODO: Test with batched data
        # TODO: Make efficient more
        # Note:
        # We must should ensure:
        # S[i,1] > 0         where i = 1..n
        # S[i,j] e (0, pi)   where i = 2..n, j = 2..i
        # We already ensure S > 0 in _chol_from_flat_sphe_chol
        # We ensure < pi by applying tanh*pi to all applicable elements
        S = sphe_chol
        n = self.action_dim
        L = th.zeros_like(sphe_chol)
        for i in range(n):
            for j in range(i):
                t = S[i, 1]
                for k in range(1, j+1):
                    t *= th.sin(th.tanh(S[i, k])*pi)
                if i != j:
                    t *= th.cos(th.tanh(S[i, j+1])*pi)
                L[i, j] = t
        return L

    def _ensure_positive_func(self, x):
        return self.enforce_positive_type.apply(x)

    def _ensure_diagonal_positive(self, chol):
        if len(chol.shape) == 1:
            # If our chol is a vector (representing a diagonal chol)
            return self._ensure_positive_func(chol)
        return chol.tril(-1) + self._ensure_positive_func(chol.diagonal(dim1=-2,
                                                                        dim2=-1)).diag_embed() + chol.triu(1)

    def string(self):
        # TODO
        return '<CholNet />'


AnyDistribution = Union[SB3_Distribution, UniversalGaussianDistribution]
