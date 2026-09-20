# Unitree G1 RL Environment

This is the early RL environment interface for Unitree G1 flat-terrain velocity
tracking. It defines observations, actions, reset behavior, stepping, and the
minimal Step 3 reward. It does not implement PPO, rollout buffers, domain
randomization, or training.

## Environment

The environment is implemented in `g1_rl_env.py` as `G1VelocityTrackingEnv`.
It uses `scene.xml`, MuJoCo, and NumPy only, so it is runnable on macOS without
CUDA, NVIDIA libraries, `mujoco-warp`, or Linux-only dependencies.

The public interface is Gymnasium-style:

```text
reset(seed=None, options=None) -> observation, info
step(action) -> observation, reward, terminated, truncated, info
```

`step()` returns the minimal velocity-tracking reward documented below and adds
the individual reward components to `info`.

## Observation Vector

Observation dimension: `96`

```text
0:29     joint position error: q - q_default
29:58    joint velocity: qd
58:61    base angular velocity
61:64    projected gravity
64:67    commanded velocity: linear x, linear y, yaw
67:96    previous action
```

Joint position error and joint velocity use the 29 actuated joints listed in the
action section. `q_default` comes from the `stand` keyframe in `g1.xml`.

Base angular velocity is taken from the `imu-pelvis-angular-velocity` gyro
sensor, which is aligned with the pelvis/body frame. Projected gravity is the
normalized MuJoCo gravity vector rotated into the pelvis/body frame. This
exposes body orientation without directly exposing a global quaternion.

The commanded velocity is deterministic by default: `[0.0, 0.0, 0.0]`. It can
be changed at construction time with `commanded_velocity=...` or during reset
with `options={"commanded_velocity": ...}`. Command curricula are intentionally
left for a later step.

## Action Vector

Action dimension: `29`

The action is a direct target joint-position vector for the existing MuJoCo
position actuators. The action ordering is exactly the loaded model actuator
ordering:

```text
0   left_hip_pitch_joint
1   left_hip_roll_joint
2   left_hip_yaw_joint
3   left_knee_joint
4   left_ankle_pitch_joint
5   left_ankle_roll_joint
6   right_hip_pitch_joint
7   right_hip_roll_joint
8   right_hip_yaw_joint
9   right_knee_joint
10  right_ankle_pitch_joint
11  right_ankle_roll_joint
12  waist_yaw_joint
13  waist_roll_joint
14  waist_pitch_joint
15  left_shoulder_pitch_joint
16  left_shoulder_roll_joint
17  left_shoulder_yaw_joint
18  left_elbow_joint
19  left_wrist_roll_joint
20  left_wrist_pitch_joint
21  left_wrist_yaw_joint
22  right_shoulder_pitch_joint
23  right_shoulder_roll_joint
24  right_shoulder_yaw_joint
25  right_elbow_joint
26  right_wrist_roll_joint
27  right_wrist_pitch_joint
28  right_wrist_yaw_joint
```

The environment validates this order against `model.actuator_trnid` at startup
and raises a clear error if the XML actuator order changes.

## Action Scaling And Clipping

There is no normalized action scaling in Step 2. Actions are desired joint
positions in radians. Before writing to `data.ctrl`, actions are clipped to the
MuJoCo actuator control ranges inherited from each joint range in the XML.

Invalid action shapes or non-finite action values raise `ValueError`.

## Default Joint Configuration

Reset uses the MuJoCo `stand` keyframe:

```text
base position/orientation from stand
joint positions from stand
controls from stand
joint and base velocities reset by MuJoCo keyframe reset
```

`previous_action` is reset to the `stand` keyframe control vector so the initial
observation is consistent with the default pose.

## Termination

Termination is intentionally minimal. An episode terminates if either:

```text
base height < 0.45 m
pelvis up-vector world z < 0.35
```

`max_episode_steps` defaults to `1000`; hitting it returns `truncated=True`.
The full task-specific episode design remains for later steps.

## Reward Function

The reward is deliberately minimal:

```text
R =
    w_v R_velocity
    + w_s R_survival
    - w_t R_torque
```

Default coefficients are defined by `RewardConfig`:

```text
w_v = 1.0
w_s = 0.1
w_t = 0.0001
sigma_linear = 0.25
sigma_yaw = 0.25
```

### Velocity Tracking

Measured linear velocity is `base_linear_velocity[:2]`, where the floating-base
linear velocity is rotated into the pelvis/body frame. Commanded linear velocity
is `commanded_velocity[:2]`.

```text
linear_error = commanded_velocity[:2] - base_linear_velocity[:2]
R_linear = exp(-(linear_error dot linear_error) / sigma_linear)
```

Measured yaw velocity is `base_angular_velocity[2]` from the
`imu-pelvis-angular-velocity` gyro sensor. Commanded yaw velocity is
`commanded_velocity[2]`.

```text
yaw_error = commanded_velocity[2] - base_angular_velocity[2]
R_yaw = exp(-(yaw_error ** 2) / sigma_yaw)
```

The combined velocity reward is:

```text
R_velocity = 0.5 * (R_linear + R_yaw)
```

### Survival

Survival is applied only while the environment is not terminated:

```text
R_survival = 1.0 if not terminated else 0.0
```

The survival contribution is therefore `0.1` per alive step with the default
configuration.

### Torque Penalty

The torque penalty uses `data.actuator_force`, MuJoCo's actuator output force
for the 29 position actuators. This is used instead of `data.ctrl` because
`data.ctrl` is the desired joint-position target, not the physical actuator
effort.

```text
R_torque = data.actuator_force dot data.actuator_force
```

The weighted torque penalty is `0.0001 * R_torque` by default.

### Reward Info

`step()` exposes these entries in `info`:

```text
reward/linear_velocity_tracking
reward/yaw_velocity_tracking
reward/velocity_tracking
reward/survival
reward/torque_penalty
reward/weighted_velocity_tracking
reward/weighted_survival
reward/weighted_torque_penalty
reward/total
```

The implementation checks reward inputs and outputs for NaN and infinity and
raises a diagnostic `FloatingPointError` if reward computation becomes
numerically invalid.

## Future Work

Later steps should collect rollouts, connect an external PPO trainer, add domain
randomization, and evaluate deterministic policies. Those are intentionally not
implemented here.
