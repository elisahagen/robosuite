
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
STEP_XY = 0.18
STEP_Z = 0.25
TOL_XY = 0.013
TOL_X = 0.01
TOL_Y = 0.013
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
    
    TARGETS = [0, 90, 180, 270]
    TOL_DEG = 10
    STEP_RAD = 0.2
    last_sign = None
    change_vel_cmd = False
    # 2) distance you want per step
    if np.linalg.norm(delta_xy) > 1e-6:
        vel_cmd = delta_xy / np.linalg.norm(delta_xy) * STEP_XY
    else:
        vel_cmd = np.zeros(2)

    step_rad = 0
    local_steps = 0
    prev_vel_cmd = np.zeros(2)
    while True:
        if stage == 2 and local_steps > 53:
            break
        
        q_cur = obs["Box_to_robot0_eef_quat"]

        if stage != 2: 
            axis, angle = quat_to_axis_angle1(q_cur)
            angle_deg = math.degrees(angle)

            if any(abs(angle_deg - tgt) <= TOL_DEG for tgt in TARGETS):
                step_rad = 0
                break 
            else: 
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
                step_rad = 0
        
        rel_pos = np.array(obs["Box_to_robot0_eef_pos"])  # [x,y,z] from bread→eef

        # if abs(rel_pos[0]) < TOL_XY:
        #     vel_cmd[0] = 0
        # if abs(rel_pos[1]) < TOL_Y:
        #     vel_cmd[1] = 0

        # if abs(rel_pos[0]) < 0.015 and abs(rel_pos[1]) < 0.015 and not change_vel_cmd:    
        #     change_vel_cmd = True
        #     vel_cmd = vel_cmd * 0.5

        if abs(rel_pos[0]) < TOL_XY and abs(rel_pos[1]) < TOL_Y:    
            break

        print(vel_cmd, rel_pos)
        action = {
            "right":         np.array([vel_cmd[0], vel_cmd[1], 0, 0, 0, step_rad]),
            "right_gripper": np.array([-1.0])
        }
        a = robot.create_action_vector(action)
        obs, rew, done, _ = env.step(a)
        data_records = save_img_info(obs, env, base_dir, cam_names, step, a, rew, done, robot, data_records)
        step += 1
        local_steps += 1

    return step, data_records


def move_z_to(env, robot, base_dir, cam_names, step, target, data_records, target_obj_height):
    """
    Phase 2: With X–Y already aligned, move just in Z until within TOL_Z.
    """
    if target is not None:
        obs, rew, done, _ = env.step(robot.create_action_vector({"right": np.zeros(6), "right_gripper": np.array([+1.0])}))  
    else: 
        obs, rew, done, _ = env.step(robot.create_action_vector({"right": np.zeros(6), "right_gripper": np.array([-1.0])}))  

    z_var = np.random.uniform(-0.0015, 0.0015)
    while True:
        
        if target is not None: 
            dz = target[2] - np.array(obs["robot0_eef_pos"])[2]
            print("dz", dz)
        else:
            dz = np.array(obs["Box_to_robot0_eef_pos"])[2]
        
        if target_obj_height > 0.025:
            tolz = TOL_Z + 0.01 + z_var
        else:
            tolz = TOL_Z + z_var

        if abs(dz) < tolz:
            print("Break")
            break

        rel_pos = np.array(obs["Box_to_robot0_eef_pos"])
        if abs(rel_pos[0]) > TOL_X or abs(rel_pos[1]) > TOL_Y:
            print(f"XY drift detected (dx={rel_pos[0]:.4f}, dy={rel_pos[1]:.4f}), realigning...")
            step, data_records = move_xy_to_obj(
                env, robot, base_dir, cam_names, step, data_records, stage=2
            )
  
        step_z = STEP_Z * np.sign(dz)
        if target is not None:
            for i in range(30): 
                action = {
                    "right":         np.array([0.0,0.0, -step_z,  0,0,0]),
                    "right_gripper": np.array([+1.0])
                }
                a = robot.create_action_vector(action)
       
                obs, rew, done, _ = env.step(a)
                data_records = save_img_info(obs, env, base_dir, cam_names, step, a, rew, done, robot, data_records)
                step += 1
                
            return step, data_records 

        else: 
            action = {
                "right":         np.array([0.0,0.0, -step_z,  0,0,0]),
                "right_gripper": np.array([-1.0])
            }
        
        a = robot.create_action_vector(action)
        obs, rew, done, _ = env.step(a)
        data_records = save_img_info(obs,env,  base_dir, cam_names, step, a, rew, done, robot, data_records)
        step += 1
        abort_if_too_many_steps(step)
        
    return step, data_records 

def quat_to_axis_angle(quat):
    eulerVec = np.zeros(3)
    qx, qy, qz, qw = quat
    xyz = quat[1:]

    sinr_cosp = 2 * (qw * qx + qy * qz)
    cosr_cosp = 1 - 2 * (qx * qx + qy * qy)
    eulerVec[0] = np.arctan2(sinr_cosp, cosr_cosp)
    # pitch (y-axis rotation)
    sinp = 2 * (qw * qy - qz * qx)
    if np.abs(sinp) >= 1:
        eulerVec[1] = np.copysign(np.pi / 2, sinp)  # use 90 degrees if out of range
    else:
        eulerVec[1] = np.arcsin(sinp)
    s = math.sqrt(max(0.0, 1 - qw*qw))
    if s < 1e-8:
        return np.array([1.0, 0.0, 0.0]), 0.0
    # yaw (z-axis rotation)
    siny_cosp = 2 * (qw * qz + qx * qy)
    cosy_cosp = 1 - 2 * (qy * qy + qz * qz)
    eulerVec[2] = np.arctan2(siny_cosp, cosy_cosp)

    return xyz/s, eulerVec[2]

def quat_to_axis_angle1(q):
    """Normalize q then return (axis, angle_rad)."""
    q = q / np.linalg.norm(q)
    xyz, w = q[1:], q[0]
    angle = 2 * math.acos(np.clip(w, -1.0, 1.0))
    s = math.sqrt(max(0.0, 1 - w*w))
    if s < 1e-8:
        return np.array([1.0, 0.0, 0.0]), 0.0
    return xyz / s, angle

def axis_angle_to_quat(axis, angle_rad):
    axis = axis / np.linalg.norm(axis)
    half = angle_rad / 2.0
    return np.array([ math.cos(half), *(axis * math.sin(half)) ])

def move_xy_to_target(env, robot, base_dir, cam_names, step, target_xy, data_records): 

    obs, rew, done, _ = env.step(robot.create_action_vector({"right": np.zeros(6), "right_gripper": np.array([+1.0])}))

    current_xy = np.array(obs["robot0_eef_pos"])[:2]
    delta_xy   = np.array(target_xy)[:2] - current_xy

    # 1) decide how many equal‐length steps of size STEP_XY you need
    dist = np.linalg.norm(delta_xy)
    num_steps = int(np.ceil(dist / STEP_XY))


    # 2) distance you want per step
    if dist > 1e-6:
        vel_cmd = delta_xy / dist * STEP_XY
    else:
        vel_cmd = np.zeros(2)
    print("delta beginning", delta_xy)
    for _ in range(num_steps):
        action = {
            "right":         np.array([vel_cmd[0], vel_cmd[1], 0, 0, 0, 0]),
            "right_gripper": np.array([+1.0])
        }
        a = robot.create_action_vector(action)
        obs, rew, done, _ = env.step(a)
        data_records = save_img_info(obs, env, base_dir, cam_names, step, a, rew, done, robot, data_records)
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
        
        data_records = save_img_info(obs, env, base_dir, cam_names, step, a, rew, done, robot, data_records)
        step += 1


    for _ in range(25):
        action = {
            "right":         np.array([0.0, 0.0, -STEP_Z, 0.0, 0.0, 0.0]),
            "right_gripper": np.array([+1.0])
        }
        a = robot.create_action_vector(action)
        obs, rew, done,_ = env.step(a)
        data_records = save_img_info(obs, env, base_dir, cam_names, step, a, rew, done, robot, data_records)
        step += 1

    action = {
        "right":         np.zeros(6),
        "right_gripper": np.array([-1.0])
    }
    for i in range(10):
        a = robot.create_action_vector(action)
        obs, rew, done, _ = env.step(a)
        data_records = save_img_info(obs, env, base_dir, cam_names, step, a, rew, done, robot, data_records)
        step += 1

    for _ in range(25):
        action = {
            "right":         np.array([0.0, 0.0, +STEP_Z, 0.0, 0.0, 0.0]),
            "right_gripper": np.array([-1.0])
        }
        a = robot.create_action_vector(action)
        obs, rew, done,_ = env.step(a)
        data_records = save_img_info(obs, env, base_dir, cam_names, step, a, rew, done, robot, data_records)
        step += 1

    return step, data_records 

def auto_pick_and_place(env, robot, write_q, base_dir, cam_names, level, target_obj, target_obj_height):
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
        target_xy = [0.1, 0.14, 0.6]
    # 1) move above object
    step, data_records = move_xy_to_obj(env, robot, base_dir, cam_names, step, data_records)
    if abort_if_too_many_steps(): return

    # 2b) align position again
    step, data_records = move_xy_to_obj(env, robot, base_dir, cam_names, step, data_records, stage=2)
    if abort_if_too_many_steps(): return

    # 3) descend to object
    step, data_records = move_z_to(env, robot, base_dir, cam_names, step, target, data_records, target_obj_height)
    if abort_if_too_many_steps(): return

    # 4) close gripper
    for i in range(10):
        
        action = {
            "right":         np.array([0,0,0, 0,0,0 ]),
            "right_gripper": np.array([+1.0]),
        }
        
        action = robot.create_action_vector(action)
        
        obs,rew, done,  _ = env.step(action)
        save_img_info(obs, env, base_dir, cam_names, step, action, rew, done, robot, data_records)
        if abort_if_too_many_steps(): return
        step += 1

    # 5) lift up 15cm
    target =  np.array([0,0,0.13])
    step, data_records = move_z_to(env, robot, base_dir, cam_names, step, target, data_records, target_obj_height)

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
    base_dir = f"/home/elisa/Documents/data/robosuite_automated/stage2/teleop_dataset_{level}_{datetime.now():%Y%m%d_%H%M%S}"
    cam_names = ["left_side_view", "right_side_view",
                 "robot0_eye_in_hand_front", "robot0_eye_in_hand_back"]

    setup_dirs(base_dir, cam_names)

    if level > 1:
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
    print(env.sim.model._body_name2id)
    for cam in cam_names:
        save_intrinsic_extrinsic(env, cam, base_dir)

    target_obj_height = env.obj_height
    robot = env.robots[0]
    

    auto_pick_and_place(env, robot, write_q, base_dir, cam_names, level, target_obj, target_obj_height)


    env.close()
    print(f"Data saved under {base_dir}")