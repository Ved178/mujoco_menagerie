"""Stable-Baselines3 PPO training entry point for Unitree G1.

This script wires the existing G1 environment into an existing PPO
implementation. It intentionally does not implement PPO or GAE itself.
"""

from __future__ import annotations

import argparse
import json
from dataclasses import asdict
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

from g1_rl_env import DomainRandomizationConfig
from g1_rl_env import FrictionRandomizationConfig
from g1_rl_env import MassRandomizationConfig
from g1_rl_env import MotorStrengthRandomizationConfig
from g1_rl_env import PushRandomizationConfig
from g1_gym_env import make_g1_gym_env


@dataclass(frozen=True)
class PPOTrainingConfig:
  total_timesteps: int = 100_000
  seed: int = 0
  learning_rate: float = 3e-4
  n_steps: int = 1024
  batch_size: int = 256
  n_epochs: int = 5
  gamma: float = 0.99
  gae_lambda: float = 0.95
  clip_range: float = 0.2
  ent_coef: float = 0.0
  vf_coef: float = 0.5
  max_grad_norm: float = 0.5
  num_envs: int = 1
  max_episode_steps: int = 1000
  checkpoint_interval: int = 25_000
  log_interval: int = 1
  run_name: str | None = None
  output_dir: str = 'runs'
  resume_checkpoint: str | None = None
  domain_randomization: bool = False


def _require_sb3():
  try:
    from stable_baselines3 import PPO
    from stable_baselines3.common.callbacks import BaseCallback
    from stable_baselines3.common.callbacks import CheckpointCallback
    from stable_baselines3.common.env_util import make_vec_env
    from stable_baselines3.common.monitor import Monitor
  except ImportError as exc:
    raise ImportError(
      'Stable-Baselines3 training dependencies are not installed in this '
      'Python environment. Install them on the training machine with: '
      'pip install stable-baselines3 gymnasium torch tensorboard'
    ) from exc
  return PPO, BaseCallback, CheckpointCallback, make_vec_env, Monitor


def build_domain_randomization_config(enabled: bool) -> DomainRandomizationConfig:
  return DomainRandomizationConfig(
    enabled=enabled,
    friction=FrictionRandomizationConfig(enabled=enabled),
    mass=MassRandomizationConfig(enabled=enabled),
    motor_strength=MotorStrengthRandomizationConfig(enabled=enabled),
    pushes=PushRandomizationConfig(enabled=enabled),
  )


def create_run_dir(config: PPOTrainingConfig) -> Path:
  output_dir = Path(config.output_dir)
  output_dir.mkdir(parents=True, exist_ok=True)
  if config.run_name:
    run_name = config.run_name
  else:
    stage = 'dr' if config.domain_randomization else 'nominal'
    run_name = f'g1_ppo_{stage}_{datetime.now().strftime("%Y%m%d_%H%M%S")}'
  run_dir = output_dir / run_name
  if run_dir.exists() and config.resume_checkpoint is None:
    raise FileExistsError(f'run directory already exists: {run_dir}')
  run_dir.mkdir(parents=True, exist_ok=True)
  return run_dir


def train(config: PPOTrainingConfig) -> Path:
  PPO, BaseCallback, CheckpointCallback, make_vec_env, Monitor = _require_sb3()
  run_dir = create_run_dir(config)
  checkpoint_dir = run_dir / 'checkpoints'
  checkpoint_dir.mkdir(exist_ok=True)
  config_path = run_dir / 'config.json'
  config_path.write_text(json.dumps(asdict(config), indent=2) + '\n')

  dr_config = build_domain_randomization_config(config.domain_randomization)

  info_keywords = (
    'reward/velocity_tracking',
    'reward/survival',
    'reward/torque_penalty',
    'reward/total',
    'action_clipped',
  )

  def env_factory():
    env = make_g1_gym_env(
      seed=config.seed,
      max_episode_steps=config.max_episode_steps,
      domain_randomization=dr_config,
    )
    return Monitor(env, info_keywords=info_keywords)

  vec_env = make_vec_env(env_factory, n_envs=config.num_envs, seed=config.seed)

  class RewardComponentCallback(BaseCallback):
    def _on_step(self) -> bool:
      infos = self.locals.get('infos', [])
      if not infos:
        return True
      keys = [
        'reward/velocity_tracking',
        'reward/survival',
        'reward/torque_penalty',
        'reward/total',
      ]
      for key in keys:
        values = [info[key] for info in infos if key in info]
        if values:
          self.logger.record(key, sum(values) / len(values))
      terminations = sum(bool(info.get('terminal_observation') is not None) for info in infos)
      if terminations:
        self.logger.record('rollout/terminal_observation_count', terminations)
      return True

  checkpoint_callback = CheckpointCallback(
    save_freq=max(config.checkpoint_interval // max(config.num_envs, 1), 1),
    save_path=str(checkpoint_dir),
    name_prefix='g1_ppo',
    save_replay_buffer=False,
    save_vecnormalize=True,
  )
  reward_callback = RewardComponentCallback()

  if config.resume_checkpoint:
    model = PPO.load(
      config.resume_checkpoint,
      env=vec_env,
      tensorboard_log=str(run_dir / 'tensorboard'),
      device='auto',
    )
  else:
    model = PPO(
      'MlpPolicy',
      vec_env,
      learning_rate=config.learning_rate,
      n_steps=config.n_steps,
      batch_size=config.batch_size,
      n_epochs=config.n_epochs,
      gamma=config.gamma,
      gae_lambda=config.gae_lambda,
      clip_range=config.clip_range,
      ent_coef=config.ent_coef,
      vf_coef=config.vf_coef,
      max_grad_norm=config.max_grad_norm,
      seed=config.seed,
      verbose=1,
      tensorboard_log=str(run_dir / 'tensorboard'),
      device='auto',
    )

  model.learn(
    total_timesteps=config.total_timesteps,
    callback=[checkpoint_callback, reward_callback],
    log_interval=config.log_interval,
    reset_num_timesteps=config.resume_checkpoint is None,
    tb_log_name='ppo',
  )
  final_path = run_dir / 'final_model.zip'
  model.save(final_path)
  vec_env.close()
  return final_path


def parse_args() -> PPOTrainingConfig:
  parser = argparse.ArgumentParser()
  parser.add_argument('--total-timesteps', type=int, default=100_000)
  parser.add_argument('--seed', type=int, default=0)
  parser.add_argument('--learning-rate', type=float, default=3e-4)
  parser.add_argument('--n-steps', type=int, default=1024)
  parser.add_argument('--batch-size', type=int, default=256)
  parser.add_argument('--n-epochs', type=int, default=5)
  parser.add_argument('--gamma', type=float, default=0.99)
  parser.add_argument('--gae-lambda', type=float, default=0.95)
  parser.add_argument('--clip-range', type=float, default=0.2)
  parser.add_argument('--ent-coef', type=float, default=0.0)
  parser.add_argument('--vf-coef', type=float, default=0.5)
  parser.add_argument('--max-grad-norm', type=float, default=0.5)
  parser.add_argument('--num-envs', type=int, default=1)
  parser.add_argument('--max-episode-steps', type=int, default=1000)
  parser.add_argument('--checkpoint-interval', type=int, default=25_000)
  parser.add_argument('--log-interval', type=int, default=1)
  parser.add_argument('--run-name', type=str, default=None)
  parser.add_argument('--output-dir', type=str, default='runs')
  parser.add_argument('--resume-checkpoint', type=str, default=None)
  parser.add_argument('--domain-randomization', action='store_true')
  args = parser.parse_args()
  return PPOTrainingConfig(**vars(args))


def main() -> None:
  final_path = train(parse_args())
  print(f'Final checkpoint: {final_path}')


if __name__ == '__main__':
  main()
