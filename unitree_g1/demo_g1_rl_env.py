"""Smoke-test and optional viewer demo for the G1 RL environment.

For interactive rendering on macOS, run:

  mjpython demo_g1_rl_env.py --viewer
"""

from __future__ import annotations

import argparse
import time

import mujoco
import numpy as np

from g1_rl_env import G1VelocityTrackingEnv


def run_smoke_test(steps: int, viewer: bool) -> None:
  env = G1VelocityTrackingEnv()
  observation, info = env.reset()

  print('Environment creation: PASS')
  print('Reset: PASS')
  print(f'Observation shape: {observation.shape}')
  print(f'Action shape: {env.action_space.shape}')
  print(f'Initial commanded velocity: {info["commanded_velocity"]}')

  action = env.action_space.sample(env.rng)
  print(f'Sampled valid action shape: {action.shape}')

  if viewer:
    import mujoco.viewer

    with mujoco.viewer.launch_passive(env.model, env.data) as viewer_handle:
      for _ in range(steps):
        action = env.action_space.sample(env.rng)
        observation, reward, terminated, truncated, info = env.step(action)
        if not np.all(np.isfinite(observation)):
          raise RuntimeError('Observation contains NaN or infinite values')
        if not np.isfinite(reward):
          raise RuntimeError(f'Reward is not finite: {reward}')
        if not np.isfinite(info['reward/total']):
          raise RuntimeError(
            f'Reward total is not finite: {info["reward/total"]}'
          )
        viewer_handle.sync()
        time.sleep(env.dt)
        if terminated or truncated:
          observation, _ = env.reset()
    render_status = 'PASS'
  else:
    reward = 0.0
    for _ in range(steps):
      action = env.action_space.sample(env.rng)
      observation, reward, terminated, truncated, info = env.step(action)
      if not np.all(np.isfinite(observation)):
        raise RuntimeError('Observation contains NaN or infinite values')
      if not np.isfinite(reward):
        raise RuntimeError(f'Reward is not finite: {reward}')
      if not np.isfinite(info['reward/total']):
        raise RuntimeError(f'Reward total is not finite: {info["reward/total"]}')
      if terminated or truncated:
        observation, _ = env.reset()
    render_status = 'SKIPPED (run with --viewer under mjpython)'

  print('100-step rollout: PASS')
  print('NaN/Inf check: PASS')
  print('Reward finite: PASS')
  print(f'Observation shape after stepping: {observation.shape}')
  print(f'MuJoCo rendering: {render_status}')


def main() -> None:
  parser = argparse.ArgumentParser()
  parser.add_argument('--steps', type=int, default=100)
  parser.add_argument('--viewer', action='store_true')
  args = parser.parse_args()

  run_smoke_test(args.steps, args.viewer)


if __name__ == '__main__':
  main()
