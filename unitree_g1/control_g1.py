import mujoco
import mujoco.viewer
import time
import sys
import termios
import tty
import select

MODEL_PATH = "scene.xml"

# ---------------------------------------------------------
# G1 joints / actuators
# ---------------------------------------------------------

JOINTS = [
    "left_hip_pitch_joint",
    "left_hip_roll_joint",
    "left_hip_yaw_joint",
    "left_knee_joint",
    "left_ankle_pitch_joint",
    "left_ankle_roll_joint",

    "right_hip_pitch_joint",
    "right_hip_roll_joint",
    "right_hip_yaw_joint",
    "right_knee_joint",
    "right_ankle_pitch_joint",
    "right_ankle_roll_joint",

    "waist_yaw_joint",
    "waist_roll_joint",
    "waist_pitch_joint",

    "left_shoulder_pitch_joint",
    "left_shoulder_roll_joint",
    "left_shoulder_yaw_joint",
    "left_elbow_joint",
    "left_wrist_roll_joint",
    "left_wrist_pitch_joint",
    "left_wrist_yaw_joint",

    "right_shoulder_pitch_joint",
    "right_shoulder_roll_joint",
    "right_shoulder_yaw_joint",
    "right_elbow_joint",
    "right_wrist_roll_joint",
    "right_wrist_pitch_joint",
    "right_wrist_yaw_joint",
]

# ---------------------------------------------------------
# Settings
# ---------------------------------------------------------

JOINT_STEP = 0.05

selected = 0

# ---------------------------------------------------------
# Keyboard
# ---------------------------------------------------------

old_settings = termios.tcgetattr(sys.stdin)
tty.setcbreak(sys.stdin.fileno())


def get_key():
    if select.select([sys.stdin], [], [], 0)[0]:
        return sys.stdin.read(1)
    return None


# ---------------------------------------------------------
# Load model
# ---------------------------------------------------------

model = mujoco.MjModel.from_xml_path(MODEL_PATH)
data = mujoco.MjData(model)

# IMPORTANT:
# Initialize the model using MuJoCo's XML-defined pose.
mujoco.mj_forward(model, data)

# Save the initial standing pose
standing_qpos = data.qpos.copy()

# Desired joint positions
desired = {}

for name in JOINTS:
    joint_id = mujoco.mj_name2id(
        model,
        mujoco.mjtObj.mjOBJ_JOINT,
        name
    )

    qpos_addr = model.jnt_qposadr[joint_id]

    desired[name] = data.qpos[qpos_addr]


# ---------------------------------------------------------
# Display
# ---------------------------------------------------------

def print_status():

    print("\033[2J\033[H")

    print("=" * 60)
    print("             UNITREE G1 JOINT CONTROLLER")
    print("=" * 60)

    print()
    print(f"Selected joint: {selected}")
    print(f"Joint: {JOINTS[selected]}")

    print()
    print("Controls")
    print("--------")
    print("[ / ]     Previous / next joint")
    print("W / S     Move joint")
    print("R         Reset selected joint")
    print("SPACE     Reset robot")
    print("Q         Quit")

    print()
    print(f"Target: {desired[JOINTS[selected]]:.3f} rad")

    print()
    print("=" * 60)


print_status()


# ---------------------------------------------------------
# Simulation
# ---------------------------------------------------------

try:

    with mujoco.viewer.launch_passive(model, data) as viewer:

        while viewer.is_running():

            # ---------------------------------------------
            # Keyboard
            # ---------------------------------------------

            key = get_key()

            if key:

                # Next joint
                if key == "]":
                    selected = (selected + 1) % len(JOINTS)
                    print_status()

                # Previous joint
                elif key == "[":
                    selected = (selected - 1) % len(JOINTS)
                    print_status()

                # Increase angle
                elif key.lower() == "w":

                    name = JOINTS[selected]

                    desired[name] += JOINT_STEP

                    print_status()

                # Decrease angle
                elif key.lower() == "s":

                    name = JOINTS[selected]

                    desired[name] -= JOINT_STEP

                    print_status()

                # Reset selected joint
                elif key.lower() == "r":

                    name = JOINTS[selected]

                    joint_id = mujoco.mj_name2id(
                        model,
                        mujoco.mjtObj.mjOBJ_JOINT,
                        name
                    )

                    qpos_addr = model.jnt_qposadr[joint_id]

                    desired[name] = standing_qpos[qpos_addr]

                    print_status()

                # Reset entire robot
                elif key == " ":

                    data.qpos[:] = standing_qpos
                    data.qvel[:] = 0

                    mujoco.mj_forward(model, data)

                    for name in JOINTS:

                        joint_id = mujoco.mj_name2id(
                            model,
                            mujoco.mjtObj.mjOBJ_JOINT,
                            name
                        )

                        qpos_addr = model.jnt_qposadr[joint_id]

                        desired[name] = standing_qpos[qpos_addr]

                    print_status()

                # Quit
                elif key.lower() == "q":
                    break

            # ---------------------------------------------
            # Position control
            # ---------------------------------------------

            for i, name in enumerate(JOINTS):

                actuator_id = i

                # Position actuator:
                # send desired joint angle directly
                data.ctrl[actuator_id] = desired[name]

            # ---------------------------------------------
            # Step
            # ---------------------------------------------

            mujoco.mj_step(model, data)

            viewer.sync()

            time.sleep(model.opt.timestep)

finally:

    termios.tcsetattr(
        sys.stdin,
        termios.TCSADRAIN,
        old_settings
    )

    print("\nController stopped.")
