"""Minimal RL environment interface for Unitree G1 velocity tracking.

This module covers the early RL plumbing for observations, actions, reset,
stepping, and a minimal Step 3 reward.  It does not implement PPO, rollout
storage, domain randomization, or training.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import mujoco
import numpy as np


G1_ACTUATED_JOINTS: tuple[str, ...] = (
  'left_hip_pitch_joint',
  'left_hip_roll_joint',
  'left_hip_yaw_joint',
  'left_knee_joint',
  'left_ankle_pitch_joint',
  'left_ankle_roll_joint',
  'right_hip_pitch_joint',
  'right_hip_roll_joint',
  'right_hip_yaw_joint',
  'right_knee_joint',
  'right_ankle_pitch_joint',
  'right_ankle_roll_joint',
  'waist_yaw_joint',
  'waist_roll_joint',
  'waist_pitch_joint',
  'left_shoulder_pitch_joint',
  'left_shoulder_roll_joint',
  'left_shoulder_yaw_joint',
  'left_elbow_joint',
  'left_wrist_roll_joint',
  'left_wrist_pitch_joint',
  'left_wrist_yaw_joint',
  'right_shoulder_pitch_joint',
  'right_shoulder_roll_joint',
  'right_shoulder_yaw_joint',
  'right_elbow_joint',
  'right_wrist_roll_joint',
  'right_wrist_pitch_joint',
  'right_wrist_yaw_joint',
)


OBSERVATION_LAYOUT: dict[str, slice] = {
  'joint_position_error': slice(0, 29),
  'joint_velocity': slice(29, 58),
  'base_angular_velocity': slice(58, 61),
  'projected_gravity': slice(61, 64),
  'commanded_velocity': slice(64, 67),
  'previous_action': slice(67, 96),
}


@dataclass(frozen=True)
class SimpleBox:
  """Small Box-like space to avoid adding Gymnasium as a hard dependency."""

  low: np.ndarray
  high: np.ndarray
  dtype: np.dtype = np.dtype(np.float64)

  @property
  def shape(self) -> tuple[int, ...]:
    return self.low.shape

  def sample(self, rng: np.random.Generator | None = None) -> np.ndarray:
    rng = np.random.default_rng() if rng is None else rng
    return rng.uniform(self.low, self.high).astype(self.dtype)


@dataclass(frozen=True)
class RewardConfig:
  """Configurable weights and scales for the minimal Step 3 reward."""

  velocity_tracking_weight: float = 1.0
  survival_weight: float = 0.1
  torque_penalty_weight: float = 1e-4
  linear_velocity_sigma: float = 0.25
  yaw_velocity_sigma: float = 0.25

  def __post_init__(self) -> None:
    values = {
      'velocity_tracking_weight': self.velocity_tracking_weight,
      'survival_weight': self.survival_weight,
      'torque_penalty_weight': self.torque_penalty_weight,
      'linear_velocity_sigma': self.linear_velocity_sigma,
      'yaw_velocity_sigma': self.yaw_velocity_sigma,
    }
    for name, value in values.items():
      if not np.isfinite(value):
        raise ValueError(f'{name} must be finite, got {value}')
    if self.linear_velocity_sigma <= 0.0:
      raise ValueError('linear_velocity_sigma must be > 0')
    if self.yaw_velocity_sigma <= 0.0:
      raise ValueError('yaw_velocity_sigma must be > 0')
    if self.torque_penalty_weight < 0.0:
      raise ValueError('torque_penalty_weight must be >= 0')


class G1VelocityTrackingEnv:
  """MuJoCo-backed G1 velocity-tracking environment skeleton.

  Actions are 29 direct target joint positions in the exact MuJoCo actuator
  order.  Observations are 96 values:

  * 29 joint position errors from the standing keyframe.
  * 29 joint velocities.
  * 3 pelvis angular velocities in the pelvis/body frame.
  * 3 gravity-vector components projected into the pelvis/body frame.
  * 3 commanded velocities: x, y, yaw.
  * 29 previous actions.
  """

  action_dim = 29
  observation_dim = 96

  def __init__(
    self,
    model_path: str | Path | None = None,
    *,
    frame_skip: int = 5,
    commanded_velocity: np.ndarray | tuple[float, float, float] | None = None,
    max_episode_steps: int | None = 1000,
    fall_height: float = 0.45,
    min_base_up_z: float = 0.35,
    reward_config: RewardConfig | None = None,
    seed: int | None = None,
  ) -> None:
    if model_path is None:
      model_path = Path(__file__).with_name('scene.xml')
    self.model_path = Path(model_path)
    self.model = mujoco.MjModel.from_xml_path(str(self.model_path))
    self.data = mujoco.MjData(self.model)
    self.frame_skip = int(frame_skip)
    if self.frame_skip < 1:
      raise ValueError(f'frame_skip must be >= 1, got {frame_skip}')

    self.rng = np.random.default_rng(seed)
    self.max_episode_steps = max_episode_steps
    self.fall_height = float(fall_height)
    self.min_base_up_z = float(min_base_up_z)
    self.reward_config = (
      RewardConfig() if reward_config is None else reward_config
    )
    self.elapsed_steps = 0

    self._validate_model()
    self._joint_ids = np.array(
      [
        mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_JOINT, name)
        for name in G1_ACTUATED_JOINTS
      ],
      dtype=int,
    )
    self._qpos_addr = self.model.jnt_qposadr[self._joint_ids]
    self._qvel_addr = self.model.jnt_dofadr[self._joint_ids]
    self._pelvis_body_id = mujoco.mj_name2id(
      self.model, mujoco.mjtObj.mjOBJ_BODY, 'pelvis'
    )
    if self._pelvis_body_id < 0:
      raise ValueError('Model does not contain required body "pelvis"')
    self._pelvis_gyro_id = mujoco.mj_name2id(
      self.model, mujoco.mjtObj.mjOBJ_SENSOR, 'imu-pelvis-angular-velocity'
    )
    if self._pelvis_gyro_id < 0:
      raise ValueError(
        'Model does not contain required sensor '
        '"imu-pelvis-angular-velocity"'
      )
    self._pelvis_gyro_adr = self.model.sensor_adr[self._pelvis_gyro_id]
    self._pelvis_gyro_dim = self.model.sensor_dim[self._pelvis_gyro_id]
    if self._pelvis_gyro_dim != 3:
      raise ValueError(
        'Expected pelvis gyro sensor to have dimension 3, '
        f'got {self._pelvis_gyro_dim}'
      )

    self._stand_key_id = mujoco.mj_name2id(
      self.model, mujoco.mjtObj.mjOBJ_KEY, 'stand'
    )
    if self._stand_key_id < 0:
      raise ValueError('Model does not contain required keyframe "stand"')

    self.default_qpos = self.model.key_qpos[self._stand_key_id].copy()
    self.default_ctrl = self.model.key_ctrl[self._stand_key_id].copy()
    self.default_joint_positions = self.default_qpos[self._qpos_addr].copy()

    self.action_low = self.model.actuator_ctrlrange[:, 0].copy()
    self.action_high = self.model.actuator_ctrlrange[:, 1].copy()
    self.action_space = SimpleBox(self.action_low, self.action_high)
    self.observation_space = SimpleBox(
      low=np.full(self.observation_dim, -np.inf),
      high=np.full(self.observation_dim, np.inf),
    )

    if commanded_velocity is None:
      commanded_velocity = np.zeros(3)
    self.commanded_velocity = self._as_command(commanded_velocity)
    self.previous_action = self.default_ctrl.copy()

    self.reset()

  @property
  def dt(self) -> float:
    return float(self.model.opt.timestep * self.frame_skip)

  @property
  def joint_names(self) -> tuple[str, ...]:
    return G1_ACTUATED_JOINTS

  def reset(
    self,
    *,
    seed: int | None = None,
    options: dict[str, Any] | None = None,
  ) -> tuple[np.ndarray, dict[str, Any]]:
    if seed is not None:
      self.rng = np.random.default_rng(seed)
    options = {} if options is None else options
    if 'commanded_velocity' in options:
      self.commanded_velocity = self._as_command(options['commanded_velocity'])

    mujoco.mj_resetDataKeyframe(self.model, self.data, self._stand_key_id)
    self.data.ctrl[:] = self.default_ctrl
    self.previous_action = self.default_ctrl.copy()
    self.elapsed_steps = 0
    mujoco.mj_forward(self.model, self.data)

    obs = self._get_observation()
    return obs, self._get_info()

  def step(
    self, action: np.ndarray | list[float] | tuple[float, ...]
  ) -> tuple[np.ndarray, float, bool, bool, dict[str, Any]]:
    action = self._validate_action(action)
    clipped_action = np.clip(action, self.action_low, self.action_high)
    self.data.ctrl[:] = clipped_action

    for _ in range(self.frame_skip):
      mujoco.mj_step(self.model, self.data)

    self.elapsed_steps += 1
    self.previous_action = clipped_action.copy()
    obs = self._get_observation()
    terminated = self._is_terminated()
    truncated = (
      self.max_episode_steps is not None
      and self.elapsed_steps >= self.max_episode_steps
    )
    reward, reward_info = self.compute_reward(terminated=terminated)
    info = self._get_info()
    info.update(reward_info)
    info['action_clipped'] = bool(np.any(clipped_action != action))

    return obs, reward, terminated, truncated, info

  def render(self) -> None:
    raise NotImplementedError(
      'Use demo_g1_rl_env.py --viewer from mjpython for interactive rendering.'
    )

  def close(self) -> None:
    pass

  def compute_reward(
    self, *, terminated: bool | None = None
  ) -> tuple[float, dict[str, float]]:
    """Computes the minimal velocity-tracking reward and its components."""

    if terminated is None:
      terminated = self._is_terminated()

    command = self._require_finite(
      'commanded_velocity', self.commanded_velocity
    )
    base_linear_velocity = self._require_finite(
      'base_linear_velocity', self.base_linear_velocity
    )
    base_angular_velocity = self._require_finite(
      'base_angular_velocity', self.base_angular_velocity
    )
    actuator_force = self._require_finite(
      'actuator_force', self.data.actuator_force
    )

    config = self.reward_config
    with np.errstate(over='raise', invalid='raise'):
      try:
        linear_error = command[:2] - base_linear_velocity[:2]
        linear_error_squared = float(np.dot(linear_error, linear_error))
        linear_tracking = float(
          np.exp(-linear_error_squared / config.linear_velocity_sigma)
        )

        yaw_error = float(command[2] - base_angular_velocity[2])
        yaw_error_squared = yaw_error * yaw_error
        yaw_tracking = float(
          np.exp(-yaw_error_squared / config.yaw_velocity_sigma)
        )

        velocity_tracking = 0.5 * (linear_tracking + yaw_tracking)
        survival = 0.0 if terminated else 1.0
        torque_penalty = float(np.dot(actuator_force, actuator_force))

        total = (
          config.velocity_tracking_weight * velocity_tracking
          + config.survival_weight * survival
          - config.torque_penalty_weight * torque_penalty
        )
      except FloatingPointError as exc:
        raise FloatingPointError(
          'Reward computation produced a numerical error'
        ) from exc

    reward_values = {
      'reward/linear_velocity_tracking': linear_tracking,
      'reward/yaw_velocity_tracking': yaw_tracking,
      'reward/velocity_tracking': velocity_tracking,
      'reward/survival': survival,
      'reward/torque_penalty': torque_penalty,
      'reward/weighted_velocity_tracking': (
        config.velocity_tracking_weight * velocity_tracking
      ),
      'reward/weighted_survival': config.survival_weight * survival,
      'reward/weighted_torque_penalty': (
        config.torque_penalty_weight * torque_penalty
      ),
      'reward/total': total,
    }
    for name, value in reward_values.items():
      if not np.isfinite(value):
        raise FloatingPointError(f'{name} is not finite: {value}')
    return total, reward_values

  def _require_finite(self, name: str, value: np.ndarray) -> np.ndarray:
    value = np.asarray(value, dtype=np.float64)
    if not np.all(np.isfinite(value)):
      raise FloatingPointError(f'{name} contains NaN or infinite values')
    return value

  def _validate_model(self) -> None:
    if self.model.nu != self.action_dim:
      raise ValueError(f'Expected 29 actuators, found {self.model.nu}')

    actuator_joint_names = []
    for actuator_id in range(self.model.nu):
      joint_id = self.model.actuator_trnid[actuator_id, 0]
      joint_name = mujoco.mj_id2name(
        self.model, mujoco.mjtObj.mjOBJ_JOINT, joint_id
      )
      actuator_joint_names.append(joint_name)

    expected = list(G1_ACTUATED_JOINTS)
    if actuator_joint_names != expected:
      raise ValueError(
        'G1 actuator ordering does not match the required joint ordering:\n'
        f'expected={expected}\nactual={actuator_joint_names}'
      )

    if not np.all(self.model.actuator_ctrllimited):
      raise ValueError('Expected all G1 actuators to have control limits')

  def _validate_action(
    self, action: np.ndarray | list[float] | tuple[float, ...]
  ) -> np.ndarray:
    action = np.asarray(action, dtype=np.float64)
    if action.shape != (self.action_dim,):
      raise ValueError(
        f'Expected action shape ({self.action_dim},), got {action.shape}'
      )
    if not np.all(np.isfinite(action)):
      raise ValueError('Action contains NaN or infinite values')
    return action

  def _get_observation(self) -> np.ndarray:
    obs = np.empty(self.observation_dim, dtype=np.float64)
    obs[OBSERVATION_LAYOUT['joint_position_error']] = (
      self.joint_positions - self.default_joint_positions
    )
    obs[OBSERVATION_LAYOUT['joint_velocity']] = self.joint_velocities
    obs[OBSERVATION_LAYOUT['base_angular_velocity']] = (
      self.base_angular_velocity
    )
    obs[OBSERVATION_LAYOUT['projected_gravity']] = self.projected_gravity
    obs[OBSERVATION_LAYOUT['commanded_velocity']] = self.commanded_velocity
    obs[OBSERVATION_LAYOUT['previous_action']] = self.previous_action
    return obs

  def _get_info(self) -> dict[str, Any]:
    return {
      'commanded_velocity': self.commanded_velocity.copy(),
      'base_linear_velocity': self.base_linear_velocity.copy(),
      'base_angular_velocity': self.base_angular_velocity.copy(),
      'joint_positions': self.joint_positions.copy(),
      'joint_velocities': self.joint_velocities.copy(),
      'ctrl': self.data.ctrl.copy(),
      'previous_action': self.previous_action.copy(),
      'elapsed_steps': self.elapsed_steps,
    }

  def _is_terminated(self) -> bool:
    return bool(
      self.base_height < self.fall_height
      or self.base_up_vector_world[2] < self.min_base_up_z
    )

  def _as_command(
    self, commanded_velocity: np.ndarray | tuple[float, float, float]
  ) -> np.ndarray:
    command = np.asarray(commanded_velocity, dtype=np.float64)
    if command.shape != (3,):
      raise ValueError(
        f'commanded_velocity must have shape (3,), got {command.shape}'
      )
    if not np.all(np.isfinite(command)):
      raise ValueError('commanded_velocity contains NaN or infinite values')
    return command

  @property
  def joint_positions(self) -> np.ndarray:
    return self.data.qpos[self._qpos_addr].copy()

  @property
  def joint_velocities(self) -> np.ndarray:
    return self.data.qvel[self._qvel_addr].copy()

  @property
  def base_height(self) -> float:
    return float(self.data.qpos[2])

  @property
  def base_rotation_matrix(self) -> np.ndarray:
    return self.data.xmat[self._pelvis_body_id].reshape(3, 3).copy()

  @property
  def base_up_vector_world(self) -> np.ndarray:
    return self.base_rotation_matrix[:, 2]

  @property
  def projected_gravity(self) -> np.ndarray:
    gravity = self.model.opt.gravity.copy()
    norm = np.linalg.norm(gravity)
    if norm == 0.0:
      return np.zeros(3)
    return self.base_rotation_matrix.T @ (gravity / norm)

  @property
  def base_linear_velocity(self) -> np.ndarray:
    return self.base_rotation_matrix.T @ self.data.qvel[0:3]

  @property
  def base_angular_velocity(self) -> np.ndarray:
    stop = self._pelvis_gyro_adr + self._pelvis_gyro_dim
    return self.data.sensordata[self._pelvis_gyro_adr:stop].copy()


if __name__ == '__main__':
  env = G1VelocityTrackingEnv()
  observation, _ = env.reset()
  print(f'observation shape: {observation.shape}')
  print(f'action shape: {env.action_space.shape}')
