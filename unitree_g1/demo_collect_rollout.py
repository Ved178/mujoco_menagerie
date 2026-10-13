"""Collect a short random-policy rollout from the Unitree G1 environment."""

from __future__ import annotations

import argparse
import time
from pathlib import Path

import numpy as np

from g1_rl_env import G1VelocityTrackingEnv
from rollout_collector import RolloutConfig
from rollout_collector import assert_batches_equal
from rollout_collector import collect_rollout
from rollout_collector import load_rollout_batch
from rollout_collector import save_rollout_batch


def main() -> None:
  parser = argparse.ArgumentParser()
  parser.add_argument('--transitions', type=int, default=100)
  parser.add_argument('--episodes', type=int, default=None)
  parser.add_argument('--seed', type=int, default=7)
  parser.add_argument('--output-dir', type=Path, default=Path('rollouts'))
  parser.add_argument('--max-episode-steps', type=int, default=None)
  parser.add_argument('--policy-type', choices=['random'], default='random')
  parser.add_argument(
    '--render',
    action='store_true',
    help='Render with MuJoCo passive viewer; on macOS run with mjpython.',
  )
  parser.add_argument(
    '--realtime',
    action=argparse.BooleanOptionalAction,
    default=True,
    help='Sleep between rendered steps so the viewer is human-watchable.',
  )
  parser.add_argument(
    '--hold-viewer',
    action='store_true',
    help='Keep the MuJoCo viewer open after collection until you close it.',
  )
  args = parser.parse_args()

  env = G1VelocityTrackingEnv(seed=args.seed)
  render_callback = None
  viewer_handle = None

  if args.render:
    import mujoco.viewer

    viewer_handle = mujoco.viewer.launch_passive(env.model, env.data)

    def render_callback() -> None:
      viewer_handle.sync()
      if args.realtime:
        time.sleep(env.dt)

  try:
    batch = collect_rollout(
      env,
      config=RolloutConfig(
        num_transitions=args.transitions,
        num_episodes=args.episodes,
        max_episode_steps=args.max_episode_steps,
        seed=args.seed,
        policy_type=args.policy_type,
      ),
      render_callback=render_callback,
    )
  finally:
    if viewer_handle is not None and args.hold_viewer:
      print('Collection finished. Close the MuJoCo viewer to exit.')
      while viewer_handle.is_running():
        viewer_handle.sync()
        time.sleep(0.05)
    if viewer_handle is not None:
      viewer_handle.close()

  total_reward = float(np.sum(batch.rewards))
  mean_reward = float(np.mean(batch.rewards))
  completed = int(np.count_nonzero(batch.terminated | batch.truncated))
  terminated = int(np.count_nonzero(batch.terminated))
  truncated = int(np.count_nonzero(batch.truncated))

  print(f'Observation dimension: {batch.obs_dim}')
  print(f'Action dimension: {batch.action_dim}')
  print(f'Transitions collected: {batch.num_transitions}')
  print(f'Total reward: {total_reward:.6f}')
  print(f'Mean reward per transition: {mean_reward:.6f}')
  print(f'Completed episodes: {completed}')
  print(f'Terminated episodes: {terminated}')
  print(f'Truncated episodes: {truncated}')
  print('Finite data check: PASS')

  path = save_rollout_batch(batch, args.output_dir)
  loaded = load_rollout_batch(path)
  assert_batches_equal(batch, loaded)
  print(f'Saved rollout: {path}')
  print('Reload verification: PASS')


if __name__ == '__main__':
  main()
