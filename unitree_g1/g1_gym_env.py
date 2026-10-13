"""Gymnasium adapter for Stable-Baselines3 training."""

from __future__ import annotations

from typing import Any

import numpy as np

from g1_rl_env import DomainRandomizationConfig
from g1_rl_env import G1VelocityTrackingEnv


def _require_gymnasium():
  try:
    import gymnasium as gym
    from gymnasium import spaces
  except ImportError as exc:
    raise ImportError(
      'Gymnasium is required for PPO training. Install training dependencies '
      'with: pip install stable-baselines3 gymnasium torch tensorboard'
    ) from exc
  return gym, spaces


class G1GymEnv:
  """Thin Gymnasium-compatible wrapper around `G1VelocityTrackingEnv`."""

  metadata = {'render_modes': []}

  def __init__(
    self,
    *,
    seed: int | None = None,
    max_episode_steps: int | None = 1000,
    domain_randomization: DomainRandomizationConfig | None = None,
  ) -> None:
    gym, spaces = _require_gymnasium()
    self._gym_env_base = gym.Env
    self.env = G1VelocityTrackingEnv(
      seed=seed,
      max_episode_steps=max_episode_steps,
      domain_randomization=domain_randomization,
    )
    self.action_space = spaces.Box(
      low=self.env.action_low.astype(np.float32),
      high=self.env.action_high.astype(np.float32),
      dtype=np.float32,
    )
    self.observation_space = spaces.Box(
      low=np.full(self.env.observation_dim, -np.inf, dtype=np.float32),
      high=np.full(self.env.observation_dim, np.inf, dtype=np.float32),
      dtype=np.float32,
    )

  def reset(
    self,
    *,
    seed: int | None = None,
    options: dict[str, Any] | None = None,
  ) -> tuple[np.ndarray, dict[str, Any]]:
    observation, info = self.env.reset(seed=seed, options=options)
    return observation.astype(np.float32), info

  def step(
    self, action: np.ndarray
  ) -> tuple[np.ndarray, float, bool, bool, dict[str, Any]]:
    observation, reward, terminated, truncated, info = self.env.step(action)
    return observation.astype(np.float32), reward, terminated, truncated, info

  def close(self) -> None:
    self.env.close()


def make_g1_gym_env(**kwargs: Any):
  _require_gymnasium()
  return G1GymEnv(**kwargs)
