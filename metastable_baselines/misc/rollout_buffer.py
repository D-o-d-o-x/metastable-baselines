from typing import Any, Dict, Optional, Type, Union, NamedTuple

import numpy as np
import torch as th
from gym import spaces

from stable_baselines3.common.buffers import RolloutBuffer
from stable_baselines3.common.vec_env import VecNormalize

# TRL requires the origina mean and covariance from the policy when the datapoint was created.
# GaussianRolloutBuffer extends the RolloutBuffer by these two fields


class GaussianRolloutBufferSamples(NamedTuple):
    observations: th.Tensor
    actions: th.Tensor
    old_values: th.Tensor
    old_log_prob: th.Tensor
    advantages: th.Tensor
    returns: th.Tensor
    means: th.Tensor
    stds: th.Tensor


class GaussianRolloutBuffer(RolloutBuffer):
    def __init__(
        self,
        buffer_size: int,
        observation_space: spaces.Space,
        action_space: spaces.Space,
        device: Union[th.device, str] = "cpu",
        gae_lambda: float = 1,
        gamma: float = 0.99,
        n_envs: int = 1,
        cov_shape=None,
    ):

        super().__init__(buffer_size, observation_space, action_space,
                         device, n_envs=n_envs, gae_lambda=gae_lambda, gamma=gamma)
        self.means, self.stds = None, None
        # TODO: Correct shape for full cov matrix
        # self.action_space.shape + self.action_space.shape

        if cov_shape == None:
            cov_shape = self.action_space.shape
        self.cov_shape = cov_shape

    def reset(self) -> None:
        self.means = np.zeros(
            (self.buffer_size, self.n_envs) + self.action_space.shape, dtype=np.float32)
        self.stds = np.zeros(
            (self.buffer_size, self.n_envs) + self.cov_shape, dtype=np.float32)
        super().reset()

    def add(
        self,
        obs: np.ndarray,
        action: np.ndarray,
        reward: np.ndarray,
        episode_start: np.ndarray,
        value: th.Tensor,
        log_prob: th.Tensor,
        mean: th.Tensor,
        std: th.Tensor,
    ) -> None:
        """
        :param obs: Observation
        :param action: Action
        :param reward:
        :param episode_start: Start of episode signal.
        :param value: estimated value of the current state
            following the current policy.
        :param log_prob: log probability of the action
            following the current policy.
        :param mean: Foo
        :param std: Bar
        """

        if len(log_prob.shape) == 0:
            # Reshape 0-d tensor to avoid error
            log_prob = log_prob.reshape(-1, 1)

        # Reshape needed when using multiple envs with discrete observations
        # as numpy cannot broadcast (n_discrete,) to (n_discrete, 1)
        if isinstance(self.observation_space, spaces.Discrete):
            obs = obs.reshape((self.n_envs,) + self.obs_shape)

        self.observations[self.pos] = np.array(obs).copy()
        self.actions[self.pos] = np.array(action).copy()
        self.rewards[self.pos] = np.array(reward).copy()
        self.episode_starts[self.pos] = np.array(episode_start).copy()
        self.values[self.pos] = value.clone().cpu().numpy().flatten()
        self.log_probs[self.pos] = log_prob.clone().cpu().numpy()
        self.means[self.pos] = mean.clone().cpu().numpy()
        self.stds[self.pos] = std.clone().cpu().numpy()
        self.pos += 1
        if self.pos == self.buffer_size:
            self.full = True

    def _get_samples(self, batch_inds: np.ndarray, env: Optional[VecNormalize] = None) -> GaussianRolloutBufferSamples:
        data = (
            self.observations[batch_inds],
            self.actions[batch_inds],
            self.values[batch_inds].flatten(),
            self.log_probs[batch_inds].flatten(),
            self.advantages[batch_inds].flatten(),
            self.returns[batch_inds].flatten(),
            self.means[batch_inds].reshape((len(batch_inds), -1)),
            self.stds[batch_inds].reshape((len(batch_inds), -1)),
        )
        return GaussianRolloutBufferSamples(*tuple(map(self.to_torch, data)))
