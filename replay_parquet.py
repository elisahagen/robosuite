import pandas as pd
import numpy as np
import time
import torch
from robosuite import load_composite_controller_config
from custom_assets.BinToBinTransfer import BinToBinTransfer  
from rot_utils import rotation_6d_to_matrix, matrix_to_quaternion, quaternion_to_euler
# === Load parquet episode ===
parquet_path = "/home/elisa/Documents/data/robosuite_automated/smooth1/conv1/s_1/data/chunk-000/episode_000000.parquet"
df = pd.read_parquet(parquet_path)

# Controller setup
ctrl_cfg = load_composite_controller_config(controller="BASIC")
cam_names = [
    "left_side_view", "right_side_view",
    "robot0_eye_in_hand_front", "robot0_eye_in_hand_back"
]

# Initialize environment
env = BinToBinTransfer(
    robots="Panda",
    target_obj="bread",
    controller_configs=ctrl_cfg,
    has_renderer=True,
    has_offscreen_renderer=False,
    use_camera_obs=False,
    camera_names=cam_names,
    control_freq=10,
    ignore_done=True,
    hard_reset=True,
    initialization_noise=None,
)

obs = env.reset()

# Optionally: set bread pose from first observation state if available
if "observation.state.pos_xyzquat_right" in df.columns:
    bread_state = np.array(df.iloc[0]["observation.state.pos_xyzquat_right"])
    print("bread_state", bread_state.shape)
    bread_pos, bread_quat = bread_state[:3], bread_state[3:]
    bread_quat = bread_quat[[3, 0, 1, 2]]  # xyzw -> wxyz if needed

    obj_body_id = env.sim.model.body_name2id("Bread_main")  
    env.sim.model.body_pos[obj_body_id] = bread_pos
    env.sim.model.body_quat[obj_body_id] = bread_quat
    env.sim.data.set_joint_qpos("Bread_joint0", np.concatenate([bread_pos, bread_quat]))
    env.sim.forward()

# === Replay loop ===
for i, row in df.iterrows():
    time.sleep(0.1)  # Slow down to visualize
    pos = np.array(row["action.pos_xyzquat_right"][:3], dtype=np.float32)

    # Extract rot6d from pos_xyzquat_right (if your parquet already stores it separately, use that instead)
    # Here I'm assuming columns exist: "action.rot6d" as a list of 6 floats
    quat = np.array(row["action.pos_xyzquat_right"][3:], dtype=np.float32)
 
    # Convert to Euler angles (requires np input, so reorder and move to CPU)
    euler_angles = []
    print("q_np", quat)
    # q_xyzw = np.array([q_np[1], q_np[2], q_np[3], q_np[0]])  # convert [w,x,y,z] → [x,y,z,w]
    euler = quaternion_to_euler(quat)
    euler_angles.append(euler)
    euler_angles = np.stack(euler_angles, axis=0) .squeeze(0)              # (H, 3)
              # (3,)

    # Gripper from action.position_normalized
    grip = np.array([row["action.position_normalized"]], dtype=np.float32)

    # Final action
    print(pos, euler, euler_angles, pos.shape, euler.shape, euler_angles.shape, grip.shape)
    env_action = np.concatenate([pos, euler_angles, grip], axis=-1)
    obs, reward, done, _ = env.step(env_action)
    print(f"Step {i}: Reward={reward}, Done={done}")

print("Replay finished.")
env.close()
