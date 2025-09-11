import json
import numpy as np
import os
from robosuite import load_composite_controller_config
from custom_assets.BinToBinTransfer import BinToBinTransfer  
from datetime import datetime
import time

base_dir = "/home/elisa/Documents/data/robosuite_automated/smooth1/1/teleop_dataset_1_20250731_200144/"  
demo_file = os.path.join(base_dir, "teleop_demo")
with open(demo_file, "r") as f:
    demo_data = json.load(f)["data"]

ctrl_cfg = load_composite_controller_config(controller="BASIC")

cam_names = ["left_side_view", "right_side_view",
             "robot0_eye_in_hand_front", "robot0_eye_in_hand_back"]

init_obs = demo_data[0]["observation"]
bread_pos = np.array(init_obs["Bread_pos"])
bread_quat = np.array(init_obs["Bread_quat"])
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
    time.sleep(0.3)
    action = np.array(step_data["action"])
    obs, reward, done, _ = env.step(action)

    print(f"Step {i}: Reward={reward}, Done={done}")

print("Replay finished.")
env.close()