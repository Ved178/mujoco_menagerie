"""Rollout collection utilities for the Unitree G1 RL environment.

This module is Step 4 only: it records environment-policy interactions for
future PPO integration.  It does not compute advantages, train a model, or
implement a rollout buffer tied to any optimizer.
"""

from __future__ import annotations

import json
from dataclasses import asdict
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any
from typing import Callable
from typing import Protocol

import numpy as np


PolicyCallable = Callable[[np.ndarray], np.ndarray]


class EnvProtocol(Protocol):
  action_space: Any
  observation_space: Any

  def reset(
    self, *, seed: int | None = None, options: dict[str, Any] | None = None
  ) -> tuple[np.ndarray, dict[str, Any]]:
    ...

  def step(
    self, action: np.ndarray
  ) -> tuple[np.ndarray, float, bool, bool, dict[str, Any]]:
    ...


@dataclass(frozen=True)
class RolloutConfig:
  """Configuration for sequential single-environment rollout collection."""

  num_transitions: int | None = 100
  num_episodes: int | None = None
  max_episode_steps: int | None = None
  seed: int | None = 0
  policy_type: str = 'random'

  def __post_init__(self) -> None:
    if self.num_transitions is None and self.num_episodes is None:
      raise ValueError('num_transitions or num_episodes must be set')
    if self.num_transitions is not None and self.num_transitions < 1:
      raise ValueError('num_transitions must be >= 1 when set')
    if self.num_episodes is not None and self.num_episodes < 1:
      raise ValueError('num_episodes must be >= 1 when set')
    if self.max_episode_steps is not None and self.max_episode_steps < 1:
      raise ValueError('max_episode_steps must be >= 1 when set')


@dataclass
class RolloutBatch:
  observations: np.ndarray
  actions: np.ndarray
  rewards: np.ndarray
  next_observations: np.ndarray
  terminated: np.ndarray
  truncated: np.ndarray
  episode_ids: np.ndarray
  timesteps: np.ndarray
  commanded_velocities: np.ndarray
  actuator_forces: np.ndarray
  controls: np.ndarray
  reward_components: np.ndarray
  termination_reasons: np.ndarray
  metadata: dict[str, Any]

  @property
  def num_transitions(self) -> int:
    return int(self.rewards.shape[0])

  @property
  def obs_dim(self) -> int:
    return int(self.observations.shape[1])

  @property
  def action_dim(self) -> int:
    return int(self.actions.shape[1])

  def validate(self) -> None:
    _validate_batch(self)


class RandomPolicy:
  """Samples finite actions uniformly inside the environment action bounds."""

  def __init__(
    self,
    action_space: Any,
    *,
    seed: int | None = None,
  ) -> None:
    self.action_space = action_space
    self.rng = np.random.default_rng(seed)
    self.low = np.asarray(action_space.low, dtype=np.float64)
    self.high = np.asarray(action_space.high, dtype=np.float64)
    if self.low.shape != self.high.shape:
      raise ValueError('action_space low/high shapes differ')
    if not np.all(np.isfinite(self.low)) or not np.all(np.isfinite(self.high)):
      raise ValueError('RandomPolicy requires finite action bounds')
    if not np.all(self.low <= self.high):
      raise ValueError('action_space lower bounds exceed upper bounds')

  def __call__(self, observation: np.ndarray) -> np.ndarray:
    del observation
    return self.rng.uniform(self.low, self.high).astype(np.float64)


def collect_rollout(
  env: EnvProtocol,
  policy: PolicyCallable | None = None,
  config: RolloutConfig | None = None,
  *,
  render_callback: Callable[[], None] | None = None,
) -> RolloutBatch:
  """Collects a sequential rollout using the environment's reset/step API."""

  config = RolloutConfig() if config is None else config
  if policy is None:
    policy = RandomPolicy(env.action_space, seed=config.seed)

  observation, reset_info = env.reset(seed=config.seed)
  observation = _as_finite_vector('reset observation', observation)
  obs_dim = observation.shape[0]
  action_dim = int(np.prod(env.action_space.shape))

  observations: list[np.ndarray] = []
  actions: list[np.ndarray] = []
  rewards: list[float] = []
  next_observations: list[np.ndarray] = []
  terminated_flags: list[bool] = []
  truncated_flags: list[bool] = []
  episode_ids: list[int] = []
  timesteps: list[int] = []
  commanded_velocities: list[np.ndarray] = []
  actuator_forces: list[np.ndarray] = []
  controls: list[np.ndarray] = []
  reward_component_rows: list[np.ndarray] = []
  termination_reasons: list[int] = []

  reward_component_names: tuple[str, ...] | None = None
  episode_id = 0
  episode_timestep = 0
  completed_episodes = 0

  while not _collection_done(
    len(rewards), completed_episodes, config=config
  ):
    action = _as_finite_vector('policy action', policy(observation))
    if action.shape != (action_dim,):
      raise ValueError(
        f'policy action shape {action.shape} does not match '
        f'expected ({action_dim},)'
      )

    next_observation, reward, terminated, truncated, info = env.step(action)
    next_observation = _as_finite_vector(
      'next observation', next_observation
    )
    if next_observation.shape != (obs_dim,):
      raise ValueError(
        f'next observation shape {next_observation.shape} does not match '
        f'expected ({obs_dim},)'
      )
    reward = _as_finite_scalar('reward', reward)

    collector_truncated = (
      config.max_episode_steps is not None
      and episode_timestep + 1 >= config.max_episode_steps
    )
    truncated = bool(truncated or collector_truncated)
    terminated = bool(terminated)

    reward_names, reward_row = _reward_components_from_info(info)
    if reward_component_names is None:
      reward_component_names = reward_names
    elif reward_component_names != reward_names:
      raise ValueError(
        'reward component keys changed during rollout: '
        f'{reward_component_names} -> {reward_names}'
      )

    observations.append(observation.copy())
    actions.append(action.copy())
    rewards.append(reward)
    next_observations.append(next_observation.copy())
    terminated_flags.append(terminated)
    truncated_flags.append(truncated)
    episode_ids.append(episode_id)
    timesteps.append(episode_timestep)
    commanded_velocities.append(
      _info_vector(info, 'commanded_velocity', expected_shape=(3,))
    )
    actuator_forces.append(
      _info_vector(info, 'actuator_force', fallback_key='ctrl')
    )
    controls.append(_info_vector(info, 'ctrl'))
    reward_component_rows.append(reward_row)
    termination_reasons.append(
      _termination_reason(terminated=terminated, truncated=truncated)
    )

    if render_callback is not None:
      render_callback()

    episode_timestep += 1
    observation = next_observation

    if terminated or truncated:
      completed_episodes += 1
      if _collection_done(
        len(rewards), completed_episodes, config=config
      ):
        break
      episode_id += 1
      episode_timestep = 0
      observation, reset_info = env.reset()
      observation = _as_finite_vector('reset observation', observation)
      if observation.shape != (obs_dim,):
        raise ValueError(
          f'reset observation shape {observation.shape} does not match '
          f'expected ({obs_dim},)'
        )
    elif (
      config.num_transitions is not None
      and len(rewards) >= config.num_transitions
    ):
      break

  del reset_info
  if reward_component_names is None:
    reward_component_names = ()

  batch = RolloutBatch(
    observations=np.asarray(observations, dtype=np.float64),
    actions=np.asarray(actions, dtype=np.float64),
    rewards=np.asarray(rewards, dtype=np.float64),
    next_observations=np.asarray(next_observations, dtype=np.float64),
    terminated=np.asarray(terminated_flags, dtype=np.bool_),
    truncated=np.asarray(truncated_flags, dtype=np.bool_),
    episode_ids=np.asarray(episode_ids, dtype=np.int64),
    timesteps=np.asarray(timesteps, dtype=np.int64),
    commanded_velocities=np.asarray(commanded_velocities, dtype=np.float64),
    actuator_forces=np.asarray(actuator_forces, dtype=np.float64),
    controls=np.asarray(controls, dtype=np.float64),
    reward_components=np.asarray(reward_component_rows, dtype=np.float64),
    termination_reasons=np.asarray(termination_reasons, dtype=np.int64),
    metadata={
      'obs_dim': obs_dim,
      'action_dim': action_dim,
      'policy_type': config.policy_type,
      'collection_config': asdict(config),
      'environment': _environment_metadata(env),
      'reward_component_names': list(reward_component_names),
      'termination_reason_codes': {
        'none': 0,
        'terminated': 1,
        'truncated': 2,
        'terminated_and_truncated': 3,
      },
    },
  )
  batch.validate()
  return batch


def save_rollout_batch(
  batch: RolloutBatch,
  output_dir: str | Path,
  *,
  filename: str | None = None,
) -> Path:
  """Saves a rollout batch as compressed NPZ without overwriting files."""

  batch.validate()
  output_dir = Path(output_dir)
  output_dir.mkdir(parents=True, exist_ok=True)
  if filename is None:
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S_%f')
    filename = f'g1_rollout_{timestamp}.npz'
  path = output_dir / filename
  if path.exists():
    raise FileExistsError(f'rollout file already exists: {path}')

  np.savez_compressed(
    path,
    observations=batch.observations,
    actions=batch.actions,
    rewards=batch.rewards,
    next_observations=batch.next_observations,
    terminated=batch.terminated,
    truncated=batch.truncated,
    episode_ids=batch.episode_ids,
    timesteps=batch.timesteps,
    commanded_velocities=batch.commanded_velocities,
    actuator_forces=batch.actuator_forces,
    controls=batch.controls,
    reward_components=batch.reward_components,
    termination_reasons=batch.termination_reasons,
    metadata=np.array(json.dumps(batch.metadata), dtype=np.str_),
  )
  return path


def load_rollout_batch(path: str | Path) -> RolloutBatch:
  """Loads and validates a rollout batch saved by `save_rollout_batch`."""

  path = Path(path)
  with np.load(path, allow_pickle=False) as data:
    required = {
      'observations',
      'actions',
      'rewards',
      'next_observations',
      'terminated',
      'truncated',
      'episode_ids',
      'timesteps',
      'commanded_velocities',
      'actuator_forces',
      'controls',
      'reward_components',
      'termination_reasons',
      'metadata',
    }
    missing = required - set(data.files)
    if missing:
      raise ValueError(f'rollout file is missing fields: {sorted(missing)}')
    metadata = json.loads(str(data['metadata']))
    batch = RolloutBatch(
      observations=data['observations'].copy(),
      actions=data['actions'].copy(),
      rewards=data['rewards'].copy(),
      next_observations=data['next_observations'].copy(),
      terminated=data['terminated'].copy(),
      truncated=data['truncated'].copy(),
      episode_ids=data['episode_ids'].copy(),
      timesteps=data['timesteps'].copy(),
      commanded_velocities=data['commanded_velocities'].copy(),
      actuator_forces=data['actuator_forces'].copy(),
      controls=data['controls'].copy(),
      reward_components=data['reward_components'].copy(),
      termination_reasons=data['termination_reasons'].copy(),
      metadata=metadata,
    )
  batch.validate()
  return batch


def assert_batches_equal(left: RolloutBatch, right: RolloutBatch) -> None:
  """Raises if two rollout batches differ in arrays or metadata."""

  left.validate()
  right.validate()
  if left.metadata != right.metadata:
    raise AssertionError('metadata differs')
  for field in (
    'observations',
    'actions',
    'rewards',
    'next_observations',
    'terminated',
    'truncated',
    'episode_ids',
    'timesteps',
    'commanded_velocities',
    'actuator_forces',
    'controls',
    'reward_components',
    'termination_reasons',
  ):
    if not np.array_equal(getattr(left, field), getattr(right, field)):
      raise AssertionError(f'{field} differs')


def _validate_batch(batch: RolloutBatch) -> None:
  n = batch.rewards.shape[0]
  if n < 1:
    raise ValueError('rollout batch is empty')
  if batch.observations.ndim != 2:
    raise ValueError('observations must have shape (N, obs_dim)')
  if batch.next_observations.shape != batch.observations.shape:
    raise ValueError('next_observations must match observations shape')
  if batch.actions.ndim != 2:
    raise ValueError('actions must have shape (N, action_dim)')

  obs_dim = batch.observations.shape[1]
  action_dim = batch.actions.shape[1]
  expected_first_dims = {
    'actions': batch.actions,
    'next_observations': batch.next_observations,
    'terminated': batch.terminated,
    'truncated': batch.truncated,
    'episode_ids': batch.episode_ids,
    'timesteps': batch.timesteps,
    'commanded_velocities': batch.commanded_velocities,
    'actuator_forces': batch.actuator_forces,
    'controls': batch.controls,
    'reward_components': batch.reward_components,
    'termination_reasons': batch.termination_reasons,
  }
  for name, array in expected_first_dims.items():
    if array.shape[0] != n:
      raise ValueError(
        f'{name} first dimension {array.shape[0]} does not match N={n}'
      )

  if batch.commanded_velocities.shape != (n, 3):
    raise ValueError('commanded_velocities must have shape (N, 3)')
  if batch.actuator_forces.ndim != 2:
    raise ValueError('actuator_forces must have shape (N, effort_dim)')
  if batch.controls.ndim != 2:
    raise ValueError('controls must have shape (N, control_dim)')
  if batch.reward_components.ndim != 2:
    raise ValueError('reward_components must have shape (N, num_components)')
  if batch.terminated.dtype != np.bool_:
    raise ValueError('terminated must be boolean')
  if batch.truncated.dtype != np.bool_:
    raise ValueError('truncated must be boolean')

  finite_fields = {
    'observations': batch.observations,
    'actions': batch.actions,
    'rewards': batch.rewards,
    'next_observations': batch.next_observations,
    'commanded_velocities': batch.commanded_velocities,
    'actuator_forces': batch.actuator_forces,
    'controls': batch.controls,
    'reward_components': batch.reward_components,
  }
  for name, array in finite_fields.items():
    if not np.all(np.isfinite(array)):
      raise ValueError(f'{name} contains NaN or infinite values')

  if np.any(batch.episode_ids < 0):
    raise ValueError('episode_ids must be non-negative')
  if np.any(batch.timesteps < 0):
    raise ValueError('timesteps must be non-negative')
  if not np.all(np.isin(batch.termination_reasons, [0, 1, 2, 3])):
    raise ValueError('termination_reasons must use codes 0, 1, 2, or 3')

  metadata_obs_dim = batch.metadata.get('obs_dim')
  metadata_action_dim = batch.metadata.get('action_dim')
  if metadata_obs_dim is not None and int(metadata_obs_dim) != obs_dim:
    raise ValueError('metadata obs_dim does not match observations')
  if metadata_action_dim is not None and int(metadata_action_dim) != action_dim:
    raise ValueError('metadata action_dim does not match actions')


def _collection_done(
  transition_count: int,
  completed_episode_count: int,
  *,
  config: RolloutConfig,
) -> bool:
  enough_transitions = (
    config.num_transitions is not None
    and transition_count >= config.num_transitions
  )
  enough_episodes = (
    config.num_episodes is not None
    and completed_episode_count >= config.num_episodes
  )
  return enough_transitions or enough_episodes


def _environment_metadata(env: EnvProtocol) -> dict[str, Any]:
  metadata = {
    'class': type(env).__name__,
    'observation_shape': list(env.observation_space.shape),
    'action_shape': list(env.action_space.shape),
    'action_low': np.asarray(env.action_space.low).tolist(),
    'action_high': np.asarray(env.action_space.high).tolist(),
  }
  for attr in (
    'model_path',
    'frame_skip',
    'max_episode_steps',
    'fall_height',
    'min_base_up_z',
    'commanded_velocity',
  ):
    if not hasattr(env, attr):
      continue
    value = getattr(env, attr)
    if isinstance(value, Path):
      metadata[attr] = str(value)
    elif isinstance(value, np.ndarray):
      metadata[attr] = value.tolist()
    else:
      metadata[attr] = value
  return metadata


def _as_finite_vector(name: str, value: Any) -> np.ndarray:
  array = np.asarray(value, dtype=np.float64)
  if array.ndim != 1:
    raise ValueError(f'{name} must be 1D, got shape {array.shape}')
  if not np.all(np.isfinite(array)):
    raise ValueError(f'{name} contains NaN or infinite values')
  return array


def _as_finite_scalar(name: str, value: Any) -> float:
  scalar = float(value)
  if not np.isfinite(scalar):
    raise ValueError(f'{name} is not finite: {scalar}')
  return scalar


def _info_vector(
  info: dict[str, Any],
  key: str,
  *,
  fallback_key: str | None = None,
  expected_shape: tuple[int, ...] | None = None,
) -> np.ndarray:
  value_key = key
  if value_key not in info and fallback_key is not None:
    value_key = fallback_key
  if value_key not in info:
    if expected_shape is None:
      return np.zeros(0, dtype=np.float64)
    return np.zeros(expected_shape, dtype=np.float64)
  array = np.asarray(info[value_key], dtype=np.float64)
  if expected_shape is not None and array.shape != expected_shape:
    raise ValueError(
      f'info[{value_key!r}] shape {array.shape} does not match '
      f'expected {expected_shape}'
    )
  if not np.all(np.isfinite(array)):
    raise ValueError(f'info[{value_key!r}] contains NaN or infinite values')
  return array.copy()


def _reward_components_from_info(
  info: dict[str, Any]
) -> tuple[tuple[str, ...], np.ndarray]:
  names = tuple(sorted(k for k in info if k.startswith('reward/')))
  values = np.asarray([float(info[name]) for name in names], dtype=np.float64)
  if not np.all(np.isfinite(values)):
    raise ValueError('reward components contain NaN or infinite values')
  return names, values


def _termination_reason(*, terminated: bool, truncated: bool) -> int:
  if terminated and truncated:
    return 3
  if terminated:
    return 1
  if truncated:
    return 2
  return 0
