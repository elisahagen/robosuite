
import numpy as np
import robosuite as suite
import os
import cv2
import argparse

import math
import datetime

from datetime import datetime
import json
from robosuite import load_composite_controller_config
from custom_assets.CustomTableArena import CustomLift
from robosuite.environments.manipulation.lift import Lift
from robosuite.environments.manipulation.pick_place import PickPlace
from robosuite.models.arenas.multi_table_arena import MultiTableArena
from custom_assets.BinToBinTransfer import BinToBinTransfer

import shutil
from robosuite.models.objects import BreadObject
from instruction_templates import (
    InstructionTemplatesLevel1,
    InstructionTemplatesLevel2,
    InstructionTemplatesLevel3,
    get_instruction
)
from action_smoother import ActionSmoother
from collections import defaultdict
from utils.utils import *
from collections import deque

STEP_SIZE = 0.3
STEP_XY = 0.3
STEP_Z = 0.25
TOL_XY = 0.01
TOL_X = 0.015
TOL_Y = 0.015
TOL_Z = 0.004

CAM_MODALITIES = ["image", "depth", "segmentation"]

def setup_dirs(base_dir, camera_names):
    os.makedirs(base_dir, exist_ok=True)
    for cam in camera_names:
        os.makedirs(os.path.join(base_dir, cam),               exist_ok=True) 
        os.makedirs(os.path.join(base_dir, cam + "_depth"),     exist_ok=True)
        os.makedirs(os.path.join(base_dir, cam + "_segmentation"), exist_ok=True)
    return

def get_ee_pose(obs):
    """Return end-effector position & quaternion (x,y,z,w)."""
    pos  = obs["robot0_eef_pos"]
    quat = obs["robot0_eef_quat"]
    cor_quat = quat[[3, 0, 1, 2]]
    return np.array(pos), np.array(cor_quat)

def abort_if_too_many_steps(step, base_dir =None):

    if step > 300:
        print(f"[auto_pick_and_place] step {step} exceeded limit, aborting")
        try:
            shutil.rmtree(base_dir)
            print(f"[auto_pick_and_place] deleted partial folder {base_dir}")
        except Exception as e:
            print(f"[auto_pick_and_place] failed to delete {base_dir}: {e}")
        return True
    return False


def move_xy_to_obj(env, robot, base_dir, cam_names, step, data_records, stage=1):
    """
    Phase 1: Hold Z constant; move in X–Y only until within TOL_XY.
    """
    obs, rew, done, _ = env.step(robot.create_action_vector({"right": np.zeros(6), "right_gripper": np.array([-1.0])}))
    oscillate_cnt = total_steps = 0
    last_sign = None


    current_xy = np.array(obs["robot0_eef_pos"])[:2]
    delta_xy   = np.array(obs["Box_pos"])[:2] - current_xy

    # 2) distance you want per step
    vel_cmd = delta_xy
    local_steps = 0
    while True:
        if stage == 2 and local_steps > 13:
            break
        
        rel_pos = np.array(obs["Box_to_robot0_eef_pos"])  # [x,y,z] from bread→eef
        if abs(rel_pos[0]) < TOL_XY and abs(rel_pos[1]) < TOL_XY:
            break

        action = {
            "right":         np.array([vel_cmd[0], vel_cmd[1], 0, 0, 0, 0]),
            "right_gripper": np.array([-1.0])
        }
        a = robot.create_action_vector(action)
        obs, rew, done, _ = env.step(a)
        data_records = save_img_info(obs, base_dir, cam_names, step, a, rew, done, robot, data_records)
        step += 1
        local_steps += 1

    # while True:
        
    #     rel_pos = np.array(obs["Bread_to_robot0_eef_pos"])  # [x,y,z] from bread→eef
    #     dx, dy, _ = rel_pos

    #     if abs(dx) < TOL_XY and abs(dy) < TOL_XY:
    #         break

    #     if last_sign is not None and sign != last_sign:
    #         oscillate_cnt += 1
    #     else:
    #         oscillate_cnt = 0
    #         total_steps  += 1
        
    #     if oscillate_cnt >= 6:
    #         break
        
    #     step_x = STEP_XY * np.sign(dx)
    #     step_y = STEP_XY * np.sign(dy)
    #     sign   = np.sign(step_x or step_y) 

    #     raw_action = {
    #         "right":         np.array([ step_x, -step_y, 0,  0,0,0 ]),
    #         "right_gripper": np.array([-1.0]),
    #     }
    #     a_raw = robot.create_action_vector(raw_action)

    #     obs, rew, done,_ = env.step(a_raw)
    #     data_records = save_img_info(obs, base_dir, cam_names, step, a_raw, rew, done, robot, data_records)

    #     step += 1
    #     abort_if_too_many_steps(step)


    return step, data_records


def move_z_to(env, robot, base_dir, cam_names, step, target, data_records):
    """
    Phase 2: With X–Y already aligned, move just in Z until within TOL_Z.
    """
    if target is not None:
        obs, rew, done, _ = env.step(robot.create_action_vector({"right": np.zeros(6), "right_gripper": np.array([+1.0])}))  
    else: 
        obs, rew, done, _ = env.step(robot.create_action_vector({"right": np.zeros(6), "right_gripper": np.array([-1.0])}))  

    while True:
        
        if target is not None: 
            dz = target[2] - np.array(obs["robot0_eef_pos"])[2]
            print("dz", dz)
        else:
            dz = np.array(obs["Box_to_robot0_eef_pos"])[2]
        

        if abs(dz) < TOL_Z:
            print("Break")
            break

        step_z = STEP_Z * np.sign(dz)
        if target is not None:
            for i in range(30): 
                action = {
                    "right":         np.array([0.0,0.0, -step_z,  0,0,0]),
                    "right_gripper": np.array([+1.0])
                }
                a = robot.create_action_vector(action)
       
                obs, rew, done, _ = env.step(a)
                data_records = save_img_info(obs, base_dir, cam_names, step, a, rew, done, robot, data_records)
                step += 1
                
            return step, data_records 

        else: 
            action = {
                "right":         np.array([0.0,0.0, -step_z,  0,0,0]),
                "right_gripper": np.array([-1.0])
            }
        
        a = robot.create_action_vector(action)
        obs, rew, done, _ = env.step(a)
        data_records = save_img_info(obs, base_dir, cam_names, step, a, rew, done, robot, data_records)
        step += 1
        abort_if_too_many_steps(step)
        
    return step, data_records 


def quat_to_axis_angle(q):
    """Normalize q then return (axis, angle_rad)."""
    q = q / np.linalg.norm(q)
    w, xyz = q[0], q[1:]
    angle = 2 * math.acos(np.clip(w, -1.0, 1.0))
    s = math.sqrt(max(0.0, 1 - w*w))
    if s < 1e-8:
        return np.array([1.0, 0.0, 0.0]), 0.0
    return xyz / s, angle

def axis_angle_to_quat(axis, angle_rad):
    axis = axis / np.linalg.norm(axis)
    half = angle_rad / 2.0
    return np.array([ math.cos(half), *(axis * math.sin(half)) ])

def rotate_to(env, robot, base_dir, cam_names, step, data_records):
    """
    Rotate end-effector around its local z-axis to align yaw to 90° (TARGET_YAW),
    stepping by STEP_ROT until within ROT_TOL radians.
    """

    obs, rew, done, _ = env.step(robot.create_action_vector({
        "right": np.zeros(6),
        "right_gripper": np.array([-1.0]),
    }))
    TARGETS = [0, 90, 180, 270]
    TOL_DEG = 10
    STEP_RAD = 0.2
    oscillation_count = 0
    last_sign = None

    while True:

        q_cur = obs["Box_to_robot0_eef_quat"]
        STEPSIZE_DEG = 2.0

        axis, angle = quat_to_axis_angle(q_cur)
        angle_deg = math.degrees(angle)

        if any(abs(angle_deg - tgt) <= TOL_DEG for tgt in TARGETS):
            print("Aligned within tolerance of a 90° increment.")
            break

        x_component = axis[0] * angle
        step_rad =  STEP_RAD if x_component > 0 else -STEP_RAD

        closest = min(TARGETS, key=lambda tgt: abs(angle_deg - tgt))
        delta_deg = (closest - angle_deg)
        delta_rad = math.radians(delta_deg)

        sign      = np.sign(delta_deg)
        step_rad  = STEP_RAD * sign

        if last_sign is not None and sign != last_sign:
            oscillate_cnt += 1
        else:
            oscillate_cnt = 0
        
        if oscillate_cnt >= 6:
            print(f"[rotate_to] aborting (osc={oscillate_cnt})")
            break
    
        action = {
            "right":         np.array([0.0, 0.0, 0.0, 0.0, 0.0, step_rad]),
            "right_gripper": np.array([-1.0])
        }
        a = robot.create_action_vector(action)
        obs, rew, done, _ = env.step(a)
        
        data_records = save_img_info(obs, base_dir, cam_names, step, a, rew, done, robot, data_records)
        step += 1
        last_sign = sign 
        abort_if_too_many_steps(step)
    return step, data_records


def move_xy_to_target(env, robot, base_dir, cam_names, step, target_xy, data_records): 

    obs, rew, done, _ = env.step(robot.create_action_vector({"right": np.zeros(6), "right_gripper": np.array([+1.0])}))

    current_xy = np.array(obs["robot0_eef_pos"])[:2]
    delta_xy   = np.array(target_xy)[:2] - current_xy

    # 1) decide how many equal‐length steps of size STEP_XY you need
    num_steps = int(np.ceil((np.linalg.norm(delta_xy) * 100))) 

    # 2) distance you want per step
    vel_cmd = delta_xy
    print("delta beginning", delta_xy)
    for _ in range(num_steps):
        action = {
            "right":         np.array([vel_cmd[0], vel_cmd[1], 0, 0, 0, 0]),
            "right_gripper": np.array([+1.0])
        }
        a = robot.create_action_vector(action)
        obs, rew, done, _ = env.step(a)
        data_records = save_img_info(obs, base_dir, cam_names, step, a, rew, done, robot, data_records)
        step += 1

    current_xy = np.array(obs["robot0_eef_pos"])[:2]
    delta_xy   = np.array(target_xy)[:2] - current_xy
    print(delta_xy, "delta_xy")

    dx = delta_xy[0]
    dy = delta_xy[1]
    while abs(dx) > TOL_X or abs(dy) > TOL_Y: 
        gx, gy = obs["robot0_eef_pos"][:2]
        dx, dy = gx - target_xy[0], gy - target_xy[1]
        print("dx", dx, dy)
        if abs(dx) < TOL_X: 
            step_x = 0
            step_y = -STEP_XY * np.sign(dy)
        elif abs(dy) < TOL_Y:
            step_y = 0
            step_x = -STEP_XY * np.sign(dx)
        else:
            step_x = -STEP_XY * np.sign(dx)
            step_y = -STEP_XY * np.sign(dy)
        action = {
            "right":         np.array([step_x, step_y, 0.0, 0.0, 0.0, 0.0]),
            "right_gripper": np.array([+1.0])
        }
        a = robot.create_action_vector(action)
        obs, rew, done,_ = env.step(a)
        
        data_records = save_img_info(obs, base_dir, cam_names, step, a, rew, done, robot, data_records)
        step += 1


    for _ in range(25):
        action = {
            "right":         np.array([0.0, 0.0, -STEP_Z, 0.0, 0.0, 0.0]),
            "right_gripper": np.array([+1.0])
        }
        a = robot.create_action_vector(action)
        obs, rew, done,_ = env.step(a)
        data_records = save_img_info(obs, base_dir, cam_names, step, a, rew, done, robot, data_records)
        step += 1

    action = {
        "right":         np.zeros(6),
        "right_gripper": np.array([-1.0])
    }
    for i in range(10):
        a = robot.create_action_vector(action)
        obs, rew, done, _ = env.step(a)
        data_records = save_img_info(obs, base_dir, cam_names, step, a, rew, done, robot, data_records)
        step += 1

    for _ in range(25):
        action = {
            "right":         np.array([0.0, 0.0, +STEP_Z, 0.0, 0.0, 0.0]),
            "right_gripper": np.array([-1.0])
        }
        a = robot.create_action_vector(action)
        obs, rew, done,_ = env.step(a)
        data_records = save_img_info(obs, base_dir, cam_names, step, a, rew, done, robot, data_records)
        step += 1

    return step, data_records 

def auto_pick_and_place(env, robot, write_q, base_dir, cam_names, level, target_obj):
    """
    1) Move above object
    2) Descend & grasp
    3) Lift
    4) Move over target bin
    5) Descend & release
    """
   
    obs = env.reset()
    #print(env.target_position)
    step = 0
    target = None
    data_records = []



    def abort_if_too_many_steps():
        nonlocal step, base_dir
        if step > 300:

            print(f"[auto_pick_and_place] step {step} exceeded limit, aborting")
            try:
                shutil.rmtree(base_dir)
                print(f"[auto_pick_and_place] deleted partial folder {base_dir}")
                
            except Exception as e:
                print(f"[auto_pick_and_place] failed to delete {base_dir}: {e}")
            return True
        return False

    ee_pos, _     = get_ee_pose(obs)
    try: 
        target_xy = env.target_position #[0.05, 0.14, 0.6] #
        if target_xy[1] < 0.2:
            target_xy[1] = target_xy[1] 
        elif target_xy[1] >= 0.2:
            target_xy[1] = target_xy[1] 
        print(f"Target position: {target_xy}")
    except:
        target_xy = [0.06, 0.16, 0.6]
    # 1) move above object
    step, data_records = move_xy_to_obj(env, robot, base_dir, cam_names, step, data_records)
    if abort_if_too_many_steps(): return

    # 2) rotate to object 
    step, data_records = rotate_to(env, robot, base_dir, cam_names, step, data_records)
    if abort_if_too_many_steps(): return

    # 2b) align position again
    step, data_records = move_xy_to_obj(env, robot, base_dir, cam_names, step, data_records, stage=2)
    if abort_if_too_many_steps(): return

    # 3) descend to object
    step, data_records = move_z_to(env, robot, base_dir, cam_names, step, target, data_records)
    if abort_if_too_many_steps(): return

    # 4) close gripper
    for i in range(10):
        
        action = {
            "right":         np.array([0,0,0, 0,0,0 ]),
            "right_gripper": np.array([+1.0]),
        }
        
        action = robot.create_action_vector(action)
        
        obs,rew, done,  _ = env.step(action)
        save_img_info(obs, base_dir, cam_names, step, action, rew, done, robot, data_records)
        if abort_if_too_many_steps(): return
        step += 1

    # 5) lift up 15cm
    target =  np.array([0,0,0.13])
    step, data_records = move_z_to(env, robot, base_dir, cam_names, step, target, data_records)

    # 6) move over bin at X=0.3, Y=0
    step, data_records = move_xy_to_target(env, robot, base_dir, cam_names, step, target_xy, data_records)

    nclass = 6
    target_object = target_obj
    save_json(os.path.join(base_dir, "teleop_demo.json"), data_records, nclass)
    instr = get_instruction(base_dir, target_object, level=level, target_position=target_xy)
    save_json(os.path.join(base_dir, "instruction.json"), {"instruction": instr}, nclass)

    print("Completed automatic pick-and-place")


if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="Automated pick-and-place teleop data collection")
    parser.add_argument("--level", type=int, default=1, choices=[1, 2, 3],
                        help="Instruction level: 1 (general), 2 (coords), 3 (natural language)")
    parser.add_argument("--object", type=str , default="box", choices=["bread", "box", "milk", "cereals"],
                        help="Instruction level: 1 (general), 2 (coords), 3 (natural language)")
    args = parser.parse_args()
    target_obj = args.object.lower()
    level = args.level
    base_dir = f"/home/elisa/Documents/data/robosuite_automated/teleop_dataset_{level}_{datetime.now():%Y%m%d_%H%M%S}"
    cam_names = ["left_side_view", "right_side_view",
                 "robot0_eye_in_hand_front", "robot0_eye_in_hand_back"]

    setup_dirs(base_dir, cam_names)

    if level >= 1:
        randomize_cubes = True
    else:
        randomize_cubes = False

    write_q = queue.Queue()
    threading.Thread(target=writer_loop, args=(write_q,), daemon=True).start()

    ctrl_cfg = load_composite_controller_config(controller="BASIC")

    # 3) build env 
    env = BinToBinTransfer(
        target_obj=target_obj,
        robots="Panda",
        controller_configs=ctrl_cfg,
        has_renderer=False,
        has_offscreen_renderer=True,
        use_camera_obs=True,
        camera_names=cam_names,
        camera_heights=480,
        camera_widths =640,
        camera_depths=True,
        camera_segmentations=["class", "class", "class", "class"],
        control_freq=10,
        ignore_done=True,
        hard_reset=True,
        randomize_cubes=randomize_cubes,
        initialization_noise=None
    )
    for cam in cam_names:
        save_intrinsic_extrinsic(env, cam, base_dir)

    robot = env.robots[0]
    
    
    # xml_path = os.path.join("/home/elisa/Documents/masterthesis/git/robosuite/robosuite/models/assets/objects/", target_obj + ".xml")

    # bread_canonical = load_canonical_mesh_from_asset(xml_path)

    # canonical_out = os.path.join(base_dir, "bread_canonical.ply")
    # bread_canonical.export(canonical_out, file_type="ply", encoding="ascii")
    # print(f"[+] wrote canonical bread mesh → {canonical_out}")

    auto_pick_and_place(env, robot, write_q, base_dir, cam_names, level, target_obj)


    env.close()
    print(f"Data saved under {base_dir}")