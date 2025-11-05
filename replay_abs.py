import pandas as pd
import numpy as np
import time
import torch
from robosuite import load_composite_controller_config
from custom_assets.BinToBinTransfer import BinToBinTransfer  
from rot_utils import rotation_6d_to_matrix, matrix_to_quaternion, quaternion_to_euler
from robosuite.utils import transform_utils as T
import json
from robosuite.controllers.parts.arm.ik import InverseKinematicsController

json_path = "/home/elisa/Documents/data/robosuite_automated/stage2/teleop_dataset_2_20251105_150043/teleop_demo"

with open(json_path, "r") as f:
    data = json.load(f)

if isinstance(data, dict) and "data" in data:
    steps = data["data"]
else:
    steps = data  # fallback: file already is a list

# Filter out entries that aren’t step dictionaries
steps = [s for s in steps if isinstance(s, dict) and "action_abs" in s]

print(f"Loaded {len(steps)} steps from {json_path}")

ctrl_cfg = load_composite_controller_config(controller="BASIC")

# The Panda arm = "right" in BASIC composite setup
# ctrl_cfg["body_parts"]["right"]["type"] = "IK_POSE"
# ctrl_cfg["body_parts"]["right"]["control_delta"] = False
# ctrl_cfg["body_parts"]["right"]["input_type"] = "absolute"
# ctrl_cfg["body_parts"]["right"]["input_ref_frame"] = "world"
cam_names = [
    "left_side_view", "right_side_view",
    "robot0_eye_in_hand_front", "robot0_eye_in_hand_back"
]

# Initialize environment
env = BinToBinTransfer(
    robots="Panda",
    target_obj="box",
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

obj_body_id = env.sim.model.body_name2id("Box_main")  


# # Optionally: set bread pose from first observation state if available
# if "observation.state.pos_xyzquat_right" in df.columns:
#     bread_state = np.array(df.iloc[0]["observation.state.pos_xyzquat_right"])
#     print("bread_state", bread_state.shape)
#     bread_pos, bread_quat = bread_state[:3], bread_state[3:]
#     bread_quat = bread_quat[[3, 0, 1, 2]]  # xyzw -> wxyz if needed

#     obj_body_id = env.sim.model.body_name2id("Bread_main")  
#     env.sim.model.body_pos[obj_body_id] = bread_pos
#     env.sim.model.body_quat[obj_body_id] = bread_quat
#     env.sim.data.set_joint_qpos("Bread_joint0", np.concatenate([bread_pos, bread_quat]))
#     env.sim.forward()

# === Replay loop ===
print(steps)
for i, row in enumerate(steps):
    print(row)
    time.sleep(0.1)  # optional slowdown for visualization

    # --- Absolute target from dataset ---
    print(row, i)
    abs_action = np.array(row["action_abs"])
    gripper_action = np.array(row["action_abs"][-1])
    target_pos = abs_action[:3]
    target_euler = abs_action[3:6]

    # --- Current EEF pose from observation ---
    eef_pos = obs["robot0_eef_pos"].copy()
    eef_quat = obs["robot0_eef_quat"].copy()
    eef_mat = T.quat2mat(eef_quat)
    eef_euler = T.mat2euler(eef_mat)

    # --- Compute relative deltas ---
    delta_pos = target_pos - eef_pos

    # Convert orientation difference robustly via quaternions
    target_mat = T.euler2mat(target_euler)
    target_quat = T.mat2quat(target_mat)
    delta_quat = T.quat_multiply(target_quat, T.quat_inverse(eef_quat))
    delta_axisangle = T.quat2axisangle(delta_quat)

    # --- Combine into full relative action ---
    rel_action = np.concatenate([delta_pos, delta_axisangle, [gripper_action]])

    # --- Step environment with relative action ---
    obs, reward, done, info = env.step(rel_action)
    env.render()

print("Replay finished.")
env.close()