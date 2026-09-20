"""Controlled reward checks for the Unitree G1 RL environment."""

from __future__ import annotations

import mujoco
import numpy as np

from g1_rl_env import G1VelocityTrackingEnv


def _set_measured_velocity(
  env: G1VelocityTrackingEnv,
  *,
  linear_xy: tuple[float, float],
  yaw: float,
) -> None:
  env.data.qvel[:] = 0.0
  env.data.qvel[0:2] = linear_xy
  env.data.qvel[5] = yaw
  mujoco.mj_forward(env.model, env.data)


def _reward_for(
  env: G1VelocityTrackingEnv,
  *,
  linear_xy: tuple[float, float],
  yaw: float,
  actuator_force: float,
  terminated: bool = False,
) -> tuple[float, dict[str, float]]:
  _set_measured_velocity(env, linear_xy=linear_xy, yaw=yaw)
  env.data.actuator_force[:] = actuator_force
  reward, info = env.compute_reward(terminated=terminated)
  if not np.isfinite(reward):
    raise AssertionError(f'reward is not finite: {reward}')
  return reward, info


def main() -> None:
  env = G1VelocityTrackingEnv(
    commanded_velocity=np.array([1.0, 0.25, 0.4]),
  )

  perfect_reward, perfect_info = _reward_for(
    env,
    linear_xy=(1.0, 0.25),
    yaw=0.4,
    actuator_force=0.0,
  )
  assert perfect_info['reward/velocity_tracking'] > 0.999
  print('Velocity tracking test: PASS')

  error_reward, error_info = _reward_for(
    env,
    linear_xy=(-1.0, -1.0),
    yaw=-1.0,
    actuator_force=0.0,
  )
  assert error_info['reward/velocity_tracking'] < (
    perfect_info['reward/velocity_tracking']
  )
  assert error_reward < perfect_reward
  print('Large velocity error test: PASS')

  low_torque_reward, _ = _reward_for(
    env,
    linear_xy=(1.0, 0.25),
    yaw=0.4,
    actuator_force=0.0,
  )
  high_torque_reward, high_torque_info = _reward_for(
    env,
    linear_xy=(1.0, 0.25),
    yaw=0.4,
    actuator_force=10.0,
  )
  assert high_torque_info['reward/torque_penalty'] > 0.0
  assert high_torque_reward < low_torque_reward
  print('Torque penalty test: PASS')

  _, alive_info = _reward_for(
    env,
    linear_xy=(1.0, 0.25),
    yaw=0.4,
    actuator_force=0.0,
    terminated=False,
  )
  assert alive_info['reward/survival'] == 1.0
  assert alive_info['reward/weighted_survival'] == env.reward_config.survival_weight
  print('Survival test: PASS')

  terminated_reward, terminated_info = _reward_for(
    env,
    linear_xy=(1.0, 0.25),
    yaw=0.4,
    actuator_force=0.0,
    terminated=True,
  )
  assert terminated_info['reward/survival'] == 0.0
  assert terminated_info['reward/weighted_survival'] == 0.0
  assert terminated_reward < perfect_reward
  print('Termination reward test: PASS')

  print('Reward finite: PASS')


if __name__ == '__main__':
  main()
