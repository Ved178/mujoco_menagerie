"""Lightweight tests for Step 4 rollout collection."""

from __future__ import annotations

from pathlib import Path

import numpy as np

from rollout_collector import RandomPolicy
from rollout_collector import RolloutConfig
from rollout_collector import assert_batches_equal
from rollout_collector import collect_rollout
from rollout_collector import load_rollout_batch
from rollout_collector import save_rollout_batch


class Box:
  def __init__(self, low, high) -> None:
    self.low = np.asarray(low, dtype=np.float64)
    self.high = np.asarray(high, dtype=np.float64)

  @property
  def shape(self) -> tuple[int, ...]:
    return self.low.shape


class ToyEnv:
  def __init__(
    self,
    *,
    obs_dim: int = 4,
    action_dim: int = 2,
    terminate_at: int | None = None,
    truncate_at: int | None = None,
    nan_on_step: int | None = None,
  ) -> None:
    self.observation_space = Box(
      low=np.full(obs_dim, -np.inf), high=np.full(obs_dim, np.inf)
    )
    self.action_space = Box(
      low=np.full(action_dim, -1.0), high=np.full(action_dim, 1.0)
    )
    self.terminate_at = terminate_at
    self.truncate_at = truncate_at
    self.nan_on_step = nan_on_step
    self.episode_step = 0
    self.reset_count = 0

  def reset(self, *, seed=None, options=None):
    del seed, options
    self.episode_step = 0
    self.reset_count += 1
    return self._obs(), {}

  def step(self, action):
    action = np.asarray(action, dtype=np.float64)
    if action.shape != self.action_space.shape:
      raise ValueError('bad action shape')
    self.episode_step += 1
    obs = self._obs()
    if self.nan_on_step == self.episode_step:
      obs[0] = np.nan
    reward = float(self.episode_step)
    terminated = (
      self.terminate_at is not None and self.episode_step >= self.terminate_at
    )
    truncated = (
      self.truncate_at is not None and self.episode_step >= self.truncate_at
    )
    info = {
      'commanded_velocity': np.array([0.1, 0.0, 0.0]),
      'actuator_force': np.full(self.action_space.shape, 0.2),
      'ctrl': action.copy(),
      'reward/total': reward,
      'reward/velocity_tracking': 1.0,
    }
    return obs, reward, terminated, truncated, info

  def _obs(self):
    return np.arange(self.observation_space.shape[0], dtype=np.float64) + (
      10.0 * self.reset_count
    ) + self.episode_step


def test_random_policy_produces_valid_reproducible_actions() -> None:
  env = ToyEnv()
  policy_a = RandomPolicy(env.action_space, seed=123)
  policy_b = RandomPolicy(env.action_space, seed=123)
  action_a = policy_a(np.zeros(env.observation_space.shape))
  action_b = policy_b(np.zeros(env.observation_space.shape))
  assert action_a.shape == env.action_space.shape
  assert np.all(action_a >= env.action_space.low)
  assert np.all(action_a <= env.action_space.high)
  assert np.array_equal(action_a, action_b)


def test_collects_requested_transition_shapes_headlessly() -> None:
  env = ToyEnv()
  batch = collect_rollout(
    env, config=RolloutConfig(num_transitions=5, seed=1)
  )
  assert batch.observations.shape == (5, 4)
  assert batch.actions.shape == (5, 2)
  assert batch.rewards.shape == (5,)
  assert batch.next_observations.shape == (5, 4)
  assert batch.commanded_velocities.shape == (5, 3)
  assert batch.actuator_forces.shape == (5, 2)
  assert batch.controls.shape == (5, 2)
  assert not np.any(batch.terminated)
  assert not np.any(batch.truncated)


def test_final_transition_and_episode_counters_are_preserved() -> None:
  env = ToyEnv(terminate_at=3)
  batch = collect_rollout(
    env, config=RolloutConfig(num_transitions=3, seed=2)
  )
  assert batch.terminated.tolist() == [False, False, True]
  assert batch.truncated.tolist() == [False, False, False]
  assert batch.episode_ids.tolist() == [0, 0, 0]
  assert batch.timesteps.tolist() == [0, 1, 2]
  assert np.array_equal(batch.next_observations[-1], env._obs())


def test_truncation_and_episode_ids_are_separate() -> None:
  env = ToyEnv(truncate_at=2)
  batch = collect_rollout(
    env, config=RolloutConfig(num_transitions=5, seed=3)
  )
  assert batch.terminated.tolist() == [False, False, False, False, False]
  assert batch.truncated.tolist() == [False, True, False, True, False]
  assert batch.episode_ids.tolist() == [0, 0, 1, 1, 2]
  assert batch.timesteps.tolist() == [0, 1, 0, 1, 0]


def test_collector_max_episode_steps_marks_truncation() -> None:
  env = ToyEnv()
  batch = collect_rollout(
    env, config=RolloutConfig(num_transitions=4, max_episode_steps=2, seed=4)
  )
  assert batch.truncated.tolist() == [False, True, False, True]
  assert batch.termination_reasons.tolist() == [0, 2, 0, 2]


def test_nan_data_are_detected() -> None:
  env = ToyEnv(nan_on_step=2)
  try:
    collect_rollout(env, config=RolloutConfig(num_transitions=3, seed=5))
  except ValueError as exc:
    assert 'NaN or infinite' in str(exc)
  else:
    raise AssertionError('expected NaN validation failure')


def test_save_and_load_preserve_arrays_and_metadata(tmp_path: Path) -> None:
  env = ToyEnv(terminate_at=2)
  batch = collect_rollout(
    env, config=RolloutConfig(num_transitions=4, seed=6)
  )
  path = save_rollout_batch(batch, tmp_path, filename='rollout.npz')
  loaded = load_rollout_batch(path)
  assert_batches_equal(batch, loaded)


def test_callable_policy_is_used() -> None:
  env = ToyEnv()

  def zero_policy(observation):
    del observation
    return np.zeros(env.action_space.shape)

  batch = collect_rollout(
    env,
    policy=zero_policy,
    config=RolloutConfig(num_transitions=3, policy_type='callable'),
  )
  assert np.array_equal(batch.actions, np.zeros((3, 2)))
  assert batch.metadata['policy_type'] == 'callable'
