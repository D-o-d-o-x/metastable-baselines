from typing import Any, Dict, List, Optional, Tuple, Union
from enum import Enum

import gym
import torch as th
from torch import nn
from torch.distributions import Normal, Independent, MultivariateNormal
from math import pi

import givens

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
from ..misc.tanhBijector import TanhBijector


class Strength(Enum):
    NONE = 0
    SCALAR = 1
    DIAG = 2
    FULL = 3


class ParametrizationType(Enum):
    NONE = 0
    CHOL = 1
    SPHERICAL_CHOL = 2
    EIGEN = 3
    EIGEN_RAW = 4


class EnforcePositiveType(Enum):
    # This need to be implemented in this ugly fashion,
    # because cloudpickle does not like more complex enums

    NONE = 0
    SOFTPLUS = 1
    ABS = 2
    RELU = 3
    LOG = 4

    def apply(self, x):
        # aaaaaa
        return [nn.Identity(), nn.Softplus(beta=1, threshold=20), th.abs, nn.ReLU(inplace=False), th.log][self.value](x)


class ProbSquashingType(Enum):
    NONE = 0
    TANH = 1

    def apply(self, x):
        return [nn.Identity(), th.tanh][self.value](x)

    def apply_inv(self, x):
        return [nn.Identity(), TanhBijector.inverse][self.value](x)


def cast_to_enum(inp, Class):
    if isinstance(inp, Enum):
        return inp
    else:
        return Class[inp]


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
            if cs == Strength.NONE:
                yield (ps, cs, EnforcePositiveType.NONE, ParametrizationType.NONE)
            else:
                for ept in allowedEPTs:
                    if cs == Strength.FULL:
                        for pt in allowedPTs:
                            if pt != ParametrizationType.NONE:
                                yield (ps, cs, ept, pt)
                    else:
                        yield (ps, cs, ept, ParametrizationType.NONE)


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

    dist_kwargs['use_sde'] = use_sde

    if isinstance(action_space, gym.spaces.Box):
        assert len(
            action_space.shape) == 1, "Error: the action space must be a vector"
        return UniversalGaussianDistribution(get_action_dim(action_space), **dist_kwargs)
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

    def __init__(self, action_dim: int, use_sde: bool = False, neural_strength: Strength = Strength.DIAG, cov_strength: Strength = Strength.DIAG, parameterization_type: ParametrizationType = ParametrizationType.NONE, enforce_positive_type: EnforcePositiveType = EnforcePositiveType.ABS, prob_squashing_type: ProbSquashingType = ProbSquashingType.NONE, epsilon=1e-3, sde_learn_features=False):
        super(UniversalGaussianDistribution, self).__init__()
        self.action_dim = action_dim
        self.par_strength = cast_to_enum(neural_strength, Strength)
        self.cov_strength = cast_to_enum(cov_strength, Strength)
        self.par_type = cast_to_enum(
            parameterization_type, ParametrizationType)
        self.enforce_positive_type = cast_to_enum(
            enforce_positive_type, EnforcePositiveType)
        self.prob_squashing_type = cast_to_enum(
            prob_squashing_type, ProbSquashingType)

        self.epsilon = epsilon

        self.distribution = None
        self.gaussian_actions = None

        self.use_sde = use_sde
        self.learn_features = sde_learn_features

        assert (self.par_type != ParametrizationType.NONE) == (
            self.cov_strength == Strength.FULL), 'You should set an ParameterizationType iff the cov-strength is full'

        if self.par_type == ParametrizationType.SPHERICAL_CHOL and self.enforce_positive_type == EnforcePositiveType.NONE:
            raise Exception(
                'You need to specify an enforce_positive_type for spherical_cholesky')

    def new_dist_like_me(self, mean: th.Tensor, chol: th.Tensor):
        p = self.distribution
        if isinstance(p, Independent):
            if p.stddev.shape != chol.shape:
                chol = th.diagonal(chol, dim1=1, dim2=2)
            np = Independent(Normal(mean, chol), 1)
        elif isinstance(p, MultivariateNormal):
            np = MultivariateNormal(mean, scale_tril=chol)
        new = UniversalGaussianDistribution(self.action_dim, use_sde=self.use_sde, neural_strength=self.par_strength, cov_strength=self.cov_strength,
                                            parameterization_type=self.par_type, enforce_positive_type=self.enforce_positive_type, prob_squashing_type=self.prob_squashing_type, epsilon=self.epsilon, sde_learn_features=self.learn_features)
        new.distribution = np

        return new

    def new_dist_like_me_from_sqrt(self, mean: th.Tensor, cov_sqrt: th.Tensor):
        chol = self._sqrt_to_chol(cov_sqrt)

        new = self.new_dist_like_me(mean, chol)

        new.cov_sqrt = cov_sqrt
        new.distribution.cov_sqrt = cov_sqrt

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

        self.latent_sde_dim = latent_sde_dim

        mean_actions = nn.Linear(latent_dim, self.action_dim)
        chol = CholNet(latent_dim, self.action_dim, std_init, self.par_strength,
                       self.cov_strength, self.par_type, self.enforce_positive_type, self.prob_squashing_type, self.epsilon)

        if self.use_sde:
            self.sample_weights(self.action_dim)

        return mean_actions, chol

    def _sqrt_to_chol(self, cov_sqrt):
        vec = self.cov_strength != Strength.FULL
        batch_dims = len(cov_sqrt.shape) - 2 + 1*vec

        if vec:
            cov_sqrt = th.diag_embed(cov_sqrt)

        if batch_dims == 0:
            cov = th.mm(cov_sqrt.mT, cov_sqrt)
            cov += th.eye(cov.shape[-1])*(self.epsilon)
        else:
            cov = th.bmm(cov_sqrt.mT, cov_sqrt)
            cov += th.eye(cov.shape[-1]).expand(cov.shape)*(self.epsilon)

        chol = th.linalg.cholesky(cov)

        if vec:
            chol = th.diagonal(chol, dim1=-2, dim2=-1)

        return chol

    def proba_distribution_from_sqrt(self, mean_actions: th.Tensor, cov_sqrt: th.Tensor, latent_pi: nn.Module) -> "UniversalGaussianDistribution":
        """
        Create the distribution given its parameters (mean, cov_sqrt)

        :param mean_actions:
        :param cov_sqrt:
        :return:
        """
        self.cov_sqrt = cov_sqrt
        chol = self._sqrt_to_chol(cov_sqrt)
        self.proba_distribution(mean_actions, chol, latent_pi)
        self.distribution.cov_sqrt = cov_sqrt
        return self

    def proba_distribution(self, mean_actions: th.Tensor, chol: th.Tensor, latent_sde: th.Tensor) -> "UniversalGaussianDistribution":
        """
        Create the distribution given its parameters (mean, chol)

        :param mean_actions:
        :param chol:
        :return:
        """
        if self.use_sde:
            self._latent_sde = latent_sde if self.learn_features else latent_sde.detach()
            # TODO: Change variance of dist to include sde-spread

        if self.cov_strength in [Strength.NONE, Strength.SCALAR, Strength.DIAG]:
            self.distribution = Independent(Normal(mean_actions, chol), 1)
        elif self.cov_strength in [Strength.FULL]:
            self.distribution = MultivariateNormal(
                mean_actions, scale_tril=chol)
        if self.distribution == None:
            raise Exception('Unable to create torch distribution')
        return self

    def log_prob(self, actions: th.Tensor, gaussian_actions: Optional[th.Tensor] = None) -> th.Tensor:
        """
        Get the log probabilities of actions according to the distribution.
        Note that you must first call the ``proba_distribution()`` method.

        :param actions:
        :return:
        """
        if self.prob_squashing_type == ProbSquashingType.NONE:
            log_prob = self.distribution.log_prob(actions)
            return log_prob

        if gaussian_actions is None:
            # It will be clipped to avoid NaN when inversing tanh
            gaussian_actions = self.prob_squashing_type.apply_inv(actions)

        log_prob = self.distribution.log_prob(gaussian_actions)

        if self.prob_squashing_type == ProbSquashingType.TANH:
            log_prob -= th.sum(th.log(1 - actions **
                               2 + self.epsilon), dim=1)
            return log_prob

        raise Exception()

    def entropy(self) -> th.Tensor:
        # TODO: This will return incorrect results when using prob-squashing
        return self.distribution.entropy()

    def sample(self) -> th.Tensor:
        if self.use_sde:
            return self._sample_sde()
        else:
            return self._sample_normal()

    def _sample_normal(self) -> th.Tensor:
        # Reparametrization trick to pass gradients
        sample = self.distribution.rsample()
        self.gaussian_actions = sample
        return self.prob_squashing_type.apply(sample)

    def _sample_sde(self) -> th.Tensor:
        noise = self.get_noise(self._latent_sde)
        actions = self.distribution.mean + noise
        self.gaussian_actions = actions
        return self.prob_squashing_type.apply(actions)

    def mode(self) -> th.Tensor:
        mode = self.distribution.mean
        self.gaussian_actions = mode
        return self.prob_squashing_type.apply(mode)

    def actions_from_params(self, mean_actions: th.Tensor, log_std: th.Tensor, deterministic: bool = False, latent_pi=None) -> th.Tensor:
        # Update the proba distribution
        self.proba_distribution(mean_actions, log_std, latent_pi=latent_pi)
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
        log_prob = self.log_prob(actions, self.gaussian_actions)
        return actions, log_prob

    def sample_weights(self, batch_size=1):
        num_dims = (self.latent_sde_dim, self.action_dim)
        self.weights_dist = Normal(th.zeros(num_dims), th.ones(num_dims))
        # Reparametrization trick to pass gradients
        self.exploration_mat = self.weights_dist.rsample()
        # Pre-compute matrices in case of parallel exploration
        self.exploration_matrices = self.weights_dist.rsample((batch_size,))

    def get_noise(self, latent_sde: th.Tensor) -> th.Tensor:
        latent_sde = latent_sde if self.learn_features else latent_sde.detach()
        latent_sde = th.nn.functional.normalize(latent_sde, dim=-1)
        # Default case: only one exploration matrix
        if len(latent_sde) == 1 or len(latent_sde) != len(self.exploration_matrices):
            chol = th.diag_embed(self.distribution.stddev)
            return (th.mm(latent_sde, self.exploration_mat) @ chol)[0]
        p = self.distribution
        if isinstance(p, th.distributions.Normal) or isinstance(p, th.distributions.Independent):
            chol = th.diag_embed(self.distribution.stddev)
        elif isinstance(p, th.distributions.MultivariateNormal):
            chol = p.scale_tril

        # Use batch matrix multiplication for efficient computation
        # (batch_size, n_features) -> (batch_size, 1, n_features)
        latent_sde = latent_sde.unsqueeze(dim=1)
        # (batch_size, 1, n_actions)
        noise = th.bmm(th.bmm(latent_sde, self.exploration_matrices), chol)
        return noise.squeeze(dim=1)


class CholNet(nn.Module):
    def __init__(self, latent_dim: int, action_dim: int, std_init: float, par_strength: Strength, cov_strength: Strength, par_type: ParametrizationType, enforce_positive_type: EnforcePositiveType, prob_squashing_type: ProbSquashingType, epsilon):
        super().__init__()
        self.latent_dim = latent_dim
        self.action_dim = action_dim

        self.par_strength = par_strength
        self.cov_strength = cov_strength
        self.par_type = par_type
        self.enforce_positive_type = enforce_positive_type
        self.prob_squashing_type = prob_squashing_type

        self.epsilon = epsilon

        self._flat_chol_len = action_dim * (action_dim + 1) // 2

        self._givens_rotator = givens.Rotation(action_dim)
        self._givens_ident = th.eye(action_dim)

        # Yes, this is ugly.
        # But I don't know how this mess could be elegantly abstracted away...

        if self.par_strength == Strength.NONE:
            if self.cov_strength == Strength.NONE:
                self.chol = th.ones(self.action_dim) * std_init
            elif self.cov_strength == Strength.SCALAR:
                self.param = nn.Parameter(
                    th.Tensor([std_init]), requires_grad=True)
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
        elif self.par_strength.value > self.cov_strength.value:
            raise Exception(
                'The parameterization can not be stronger than the actual covariance.')
        else:
            if self.par_strength == Strength.SCALAR and self.cov_strength == Strength.DIAG:
                self.factor = nn.Linear(latent_dim, 1)
                self.param = nn.Parameter(
                    th.ones(self.action_dim), requires_grad=True)
            elif self.par_strength == Strength.DIAG and self.cov_strength == Strength.FULL:
                if self.enforce_positive_type == EnforcePositiveType.NONE:
                    raise Exception(
                        'For Hybrid[Diag=>Full] enforce_positive_type has to be not NONE. Otherwise required SPD-contraint can not be ensured for cov.')
                self.stds = nn.Linear(latent_dim, self.action_dim)
                self.padder = th.nn.ZeroPad2d((0, 1, 1, 0))
                # TODO: Init Non-zero?
                self.params = nn.Parameter(
                    th.ones(self._full_params_len - self.action_dim) * 0, requires_grad=True)
            elif self.par_strength == Strength.SCALAR and self.cov_strength == Strength.FULL:
                self.factor = nn.Linear(latent_dim, 1)
                # TODO: Init Off-axis differently?
                self.params = nn.Parameter(
                    th.ones(self._full_params_len) * std_init, requires_grad=True)
            else:
                raise Exception("This Exception can't happen")

    def forward(self, x: th.Tensor) -> th.Tensor:
        # Ugly mess pt.2:
        if self.par_strength == Strength.NONE:
            if self.cov_strength == Strength.NONE:
                return self.chol
            elif self.cov_strength == Strength.SCALAR:
                return self._ensure_positive_func(
                    th.ones(self.action_dim) * self.param[0])
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
                factor = self.factor(x)[0]
                diag_chol = self._ensure_positive_func(
                    self.param * factor)
                return diag_chol
            elif self.par_strength == Strength.DIAG and self.cov_strength == Strength.FULL:
                # TODO: Maybe possible to improve speed and stability by making conversion from pearson correlation + stds to cov in cholesky-form.
                stds = self._ensure_positive_func(self.stds(x))
                smol = self._parameterize_full(self.params)
                big = self.padder(smol)
                pearson_cor_chol = big + th.eye(stds.shape[-1])
                pearson_cor = (pearson_cor_chol.T @
                               pearson_cor_chol)
                if len(stds.shape) > 1:
                    # batched operation, we need to expand
                    pearson_cor = pearson_cor.expand(
                        (stds.shape[0],)+pearson_cor.shape)
                    stds = stds.unsqueeze(2)
                cov = stds.mT * pearson_cor * stds
                chol = th.linalg.cholesky(cov)
                return chol
            elif self.par_strength == Strength.SCALAR and self.cov_strength == Strength.FULL:
                # TODO: Maybe possible to improve speed and stability by multiplying with factor in cholesky-form.
                factor = self._ensure_positive_func(self.factor(x))
                par_chol = self._parameterize_full(self.params)
                cov = (par_chol.T @ par_chol)
                if len(factor) > 1:
                    factor = factor.unsqueeze(2)
                cov = cov * factor
                chol = th.linalg.cholesky(cov)
                return chol
        raise Exception()

    @property
    def _full_params_len(self):
        if self.par_type == ParametrizationType.CHOL:
            return self._flat_chol_len
        elif self.par_type == ParametrizationType.SPHERICAL_CHOL:
            return self._flat_chol_len
        elif self.par_type == ParametrizationType.EIGEN:
            return self.action_dim * 2
        elif self.par_type == ParametrizationType.EIGEN_BIJECT:
            return self.action_dim * 2
        raise Exception()

    def _parameterize_full(self, params):
        if self.par_type == ParametrizationType.CHOL:
            return self._chol_from_flat(params)
        elif self.par_type == ParametrizationType.SPHERICAL_CHOL:
            return self._chol_from_flat_sphe_chol(params)
        elif self.par_type == ParametrizationType.EIGEN:
            return self._chol_from_givens_params(params, True)
        elif self.par_type == ParametrizationType.EIGEN_RAW:
            return self._chol_from_givens_params(params, False)
        raise Exception()

    def _chol_from_flat(self, flat_chol):
        chol = fill_triangular(flat_chol)
        return self._ensure_diagonal_positive(chol)

    def _chol_from_flat_sphe_chol(self, flat_sphe_chol):
        pos_flat_sphe_chol = self._ensure_positive_func(flat_sphe_chol)
        sphe_chol = fill_triangular(pos_flat_sphe_chol)
        chol = self._chol_from_sphe_chol(sphe_chol)
        return chol

    def _chol_from_sphe_chol(self, sphe_chol):
        # TODO: Make efficient more
        # Note:
        # We must should ensure:
        # S[i,1] > 0         where i = 1..n
        # S[i,j] e (0, pi)   where i = 2..n, j = 2..i
        # We already ensure S > 0 in _chol_from_flat_sphe_chol
        # We ensure < pi by applying tanh*pi to all applicable elements
        vec = self.cov_strength != Strength.FULL
        batch_dims = len(sphe_chol.shape) - 2 + 1*vec
        batch = batch_dims != 0
        batch_shape = sphe_chol.shape[:batch_dims]
        batch_shape_scalar = batch_shape + (1,)

        S = sphe_chol
        n = sphe_chol.shape[-1]
        L = th.zeros_like(sphe_chol)
        for i in range(n):
            #t = 1
            t = th.Tensor([1])[0]
            if batch:
                t = t.expand(batch_shape_scalar)
            #s = ''
            for j in range(i+1):
                #maybe_cos = 1
                maybe_cos = th.Tensor([1])[0]
                if batch:
                    maybe_cos = maybe_cos.expand(batch_shape_scalar)
                #s_maybe_cos = ''
                if i != j and j < n-1 and i < n:
                    if batch:
                        maybe_cos = th.cos(th.tanh(S[:, i, j+1])*pi)
                    else:
                        maybe_cos = th.cos(th.tanh(S[i, j+1])*pi)
                    #s_maybe_cos = 'cos([l_'+str(i+1)+']_'+str(j+2)+')'
                if batch:
                    # try:
                    L[:, i, j] = (S[:, i, 0] * t.T) * maybe_cos.T
                    # except:
                    #    import pdb
                    #    pdb.set_trace()
                else:
                    L[i, j] = S[i, 0] * t * maybe_cos
                # print('[L_'+str(i+1)+']_'+str(j+1) +
                #      '=[l_'+str(i+1)+']_1'+s+s_maybe_cos)
                if j <= i and j < n-1 and i < n:
                    if batch:
                        tc = t.clone()
                        t = (tc.T * th.sin(th.tanh(S[:, i, j+1])*pi)).T
                    else:
                        t *= th.sin(th.tanh(S[i, j+1])*pi)
                    #s += 'sin([l_'+str(i+1)+']_'+str(j+2)+')'
        return L

    def _ensure_positive_func(self, x):
        return self.enforce_positive_type.apply(x) + self.epsilon

    def _ensure_diagonal_positive(self, chol):
        if len(chol.shape) == 1:
            # If our chol is a vector (representing a diagonal chol)
            return self._ensure_positive_func(chol)
        return chol.tril(-1) + self._ensure_positive_func(chol.diagonal(dim1=-2,
                                                                        dim2=-1)).diag_embed() + chol.triu(1)

    def _chol_from_givens_params(self, params, bijection=False):
        theta, eigenv = params[:self.action_dim], params[self.action_dim:]

        eigenv = self._ensure_positive_func(eigenv)

        if bijection:
            eigenv = th.cumsum(eigenv, -1)
            # reverse order, oh well...

        self._givens_rot.theta = theta
        Q = self._givens_rotator(self._givens_ident)
        Qinv = Q.transpose(dim0=-2, dim1=-1)

        cov = Q * th.diag(eigenv) * Qinv
        chol = th.linalg.cholesky(cov)

        return chol

    def string(self):
        return '<CholNet />'


AnyDistribution = Union[SB3_Distribution, UniversalGaussianDistribution]
