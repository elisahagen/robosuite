import json
import numpy as np
import os
from robosuite import load_composite_controller_config
from custom_assets.BinToBinTransfer import BinToBinTransfer  
from datetime import datetime
import time
from robosuite.utils import transform_utils as T

base_dir = "/home/elisa/Documents/data/robosuite_automated/smoothrot/s_2/teleop_dataset_1_20250903_171821/"  
demo_file = os.path.join(base_dir, "teleop_demo")
with open(demo_file, "r") as f:
    demo_data = json.load(f)["data"]

ctrl_cfg = load_composite_controller_config(controller="BASIC")
print("Available body parts:", ctrl_cfg["body_parts"]["right"].keys())
ctrl_cfg["body_parts"]["right"]["input_type"] = "absolute"  # for JointPosition-based parts
ctrl_cfg["body_parts"]["right"]["control_delta"] = False     # for IK-based parts
ctrl_cfg["body_parts"]["right"]["input_ref_frame"] = "world"

cam_names = ["left_side_view", "right_side_view",
             "robot0_eye_in_hand_front", "robot0_eye_in_hand_back"]

init_obs = demo_data[0]["observation"]
bread_pos = np.array(init_obs["Box_pos"])
bread_quat = np.array(init_obs["Box_quat"])
bread_quat = bread_quat[[3, 0, 1, 2]] 

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

env.sim.model.body_pos[obj_body_id] = bread_pos
env.sim.model.body_quat[obj_body_id] = bread_quat  

joint_name = "Box_joint0"  
env.sim.data.set_joint_qpos(joint_name, np.concatenate([bread_pos, bread_quat]))
env.sim.forward()

for i, step_data in enumerate(demo_data):
    time.sleep(0.1)
    action = np.array(step_data["action"])
    eef_pos = obs["robot0_eef_pos"].copy()
    eef_quat = obs["robot0_eef_quat"].copy()

    print("action", action, "\n eef", eef_pos)
    # --- Compute absolute target pose (as robosuite controller does internally) ---
    delta_pos = action[:3]
    delta_rot_axisangle = action[3:6]

    # Convert delta rotation to quaternion and combine
    delta_quat = T.axisangle2quat(delta_rot_axisangle)
    target_quat = T.quat_multiply(delta_quat, eef_quat)
    target_pos = eef_pos + delta_pos

    # Controller expects axis-angle, not quaternion
    target_axisangle = np.zeros(3) #T.quat2axisangle(target_quat)
    abs_action = np.hstack((target_pos, target_axisangle))

    # Build composite controller input
    all_action = {
        "arm": np.hstack((target_pos, target_axisangle)),  # Absolute 6D pose
        "gripper": np.array([action[-1]]),                 # Gripper action
    }
    # Flatten for robosuite
    action_vec = action #env.robots[0].create_action_vector(action)

    # Apply to environment
    obs, reward, done, info = env.step(action_vec)
    # env.render()
    # env.sim.forward()
    env.render()

print("Replay finished.")
env.close()