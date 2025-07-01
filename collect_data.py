import numpy as np
import robosuite as suite
import os
import cv2
import argparse
import threading
import queue
import math
import datetime
import random
from datetime import datetime
import json
from robosuite import load_composite_controller_config
from custom_assets.CustomTableArena import CustomLift
from robosuite.environments.manipulation.lift import Lift
from robosuite.environments.manipulation.pick_place import PickPlace
from robosuite.models.arenas.multi_table_arena import MultiTableArena
from custom_assets.BinToBinTransfer import BinToBinTransfer
from scipy.spatial.transform import Rotation as R
import shutil
import trimesh
from robosuite.models.objects import BreadObject
import xml.etree.ElementTree as ET

INSTRUCTION_TEMPLATES = {
"milk": [
        "Pick up the milk and place it into the empty space in the bin.",
        "Grab the milk carton and move it to the unoccupied spot in the bin.",
        "Lift the milk and carefully place it in the free location among the cubes.",
        "Take the milk and put it into the vacant slot inside the bin."
    ],
    "bread": [
        "Pick up the bread and place it in the available empty space in the bin.",
        "Grab the loaf of bread and move it into the unfilled position.",
        "Lift the bread and drop it in the remaining free space next to the cubes.",
        "Relocate the bread to the empty slot inside the target bin area."
    ],
    "can": [
        "Take the can and place it into the only remaining empty space in the bin.",
        "Pick up the can and move it into the free spot among the cubes.",
        "Grab the can and place it into the vacant location left in the bin.",
        "Relocate the can into the unoccupied space on the table."
    ],
    "juice": [
        "Lift the juice bottle and set it into the empty space in the bin.",
        "Grab the juice container and drop it into the last available slot.",
        "Take the juice and place it in the unoccupied area next to the cubes."
    ],
    "cereals": [
        "Lift the cereal box and place it into the remaining empty spot in the bin.",
        "Grab the cereal box and drop it into the open space near the cubes.",
        "Take the cereal box and set it in the only free position inside the bin."
    ],
    "bottle": [
        "Lift the bottle and place it into the bin's remaining empty space.",
        "Grab the bottle and drop it into the free slot between the cubes.",
        "Take the bottle and put it into the unoccupied spot in the bin."
    ],
    "cube": [
        "Lift the cube and set it into the last empty space in the bin.",
        "Grab the cube and place it into the unfilled slot among the other cubes.",
        "Take the cube and drop it into the open area left in the bin."
    ],
    "box": [
        "Lift the box and place it into the free space in the bin.",
        "Grab the box and carefully set it in the only available spot.",
        "Take the box and position it in the bin where there is no other cube."
    ],
    "capsule": [
        "Lift the capsule and place it into the empty position inside the bin.",
        "Grab the capsule and drop it into the free area left between the cubes.",
        "Take the capsule and put it into the remaining unoccupied space."
    ],
    "cylinder": [
        "Lift the cylinder and set it into the last free space in the bin.",
        "Grab the cylinder and move it into the available slot near the other cubes.",
        "Take the cylinder and place it into the bin's empty location."
    ]
}

STEP_SIZE = 0.3
UP_DOWN   = 0.25
HOME_JOINTS = [0., np.pi/4., 0., -np.pi/4., 0., np.pi/2., 0.]

STEP_XY = 0.3
STEP_Z = 0.25
STEP_ROT = 0.07 
STEP_TIL = 0.08
TOL_TIL = 0.025
TOL_XY = 0.005
TOL_Z = 0.004

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

def load_canonical_mesh_from_asset(xml_path):
    txt = open(xml_path, 'r').read()
    wrapped = "<root>\n" + txt + "\n</root>"
    root = ET.fromstring(wrapped)

    # 2) find the first <mesh> anywhere under it
    mesh_elem = root.find(".//mesh")
    mesh_file = mesh_elem.attrib["file"]
    scale_str = mesh_elem.attrib.get("scale", "1 1 1")
    scales = [float(s) for s in scale_str.split()]

    # 3) resolve path and load+scale
    mesh_path = os.path.join(os.path.dirname(xml_path), mesh_file)
    mesh = trimesh.load(mesh_path, force="mesh")
    mesh.apply_scale(scales)
    return mesh

def convert_obs(obs):
    """
    Convert observations to JSON‐serializable format.
    """
    if isinstance(obs, np.ndarray):
        return obs.tolist()
    elif isinstance(obs, dict):
        return {key: convert_obs(val) for key, val in obs.items()}
    elif isinstance(obs, list):
        return [convert_obs(elem) for elem in obs]
    else:
        return obs
    
def save_intrinsic_extrinsic(env, cam, base_dir):
    cam_id = env.sim.model.camera_name2id(cam)

    # --- Intrinsics ---
    fovy = env.sim.model.cam_fovy[cam_id]
    width = env.camera_widths[env.camera_names.index(cam)]
    height = env.camera_heights[env.camera_names.index(cam)]
    # focal_length = 0.5 * height / np.tan(np.deg2rad(fovy / 2))

    fovy_rad = np.deg2rad(fovy)
    fy = 0.5 * height / np.tan(fovy_rad / 2)
    fx = fy * (width / height)

    intrinsics = {
        "camera_name": cam,
        "image_width": width,
        "image_height": height,
        "fovy_deg": fovy,
        "focal_length_px": {"fx": fx, "fy": fy},
        "intrinsic_matrix_K": [
            [fx, 0, width / 2],
            [0, fy, height / 2],
            [0, 0, 1]
        ]
    }

    # --- Extrinsics ---
    cam_pos = env.sim.model.cam_pos[cam_id].tolist()
    cam_quat = env.sim.model.cam_quat[cam_id]  # [x, y, z, w]
    
    rotation_matrix = R.from_quat(cam_quat).as_matrix().tolist()
    cam_quat = [cam_quat[1], cam_quat[2], cam_quat[3], cam_quat[0]]
    
    extrinsics = {
        "position_xyz": cam_pos,
        "rotation_quaternion_wxyz": cam_quat,
        "rotation_matrix": rotation_matrix,
        "camera_to_world_matrix": [
            rotation_matrix[0] + [cam_pos[0]],
            rotation_matrix[1] + [cam_pos[1]],
            rotation_matrix[2] + [cam_pos[2]],
            [0, 0, 0, 1]
        ]
    }

    # Combine both
    full_cam_info = {
        "camera_name": cam,
        "intrinsics": intrinsics,
        "extrinsics": extrinsics
    }

    save_path = os.path.join(base_dir, f"{cam}_camera_info.json")
    with open(save_path, "w") as f_out:
        json.dump(full_cam_info, f_out, indent=2)

    print(f"Saved camera info for {cam} to {save_path}")

CAM_MODALITIES = ["image", "depth", "segmentation"]


def setup_dirs(base_dir, camera_names):
    os.makedirs(base_dir, exist_ok=True)
    for cam in camera_names:
        os.makedirs(os.path.join(base_dir, cam),               exist_ok=True) 
        os.makedirs(os.path.join(base_dir, cam + "_depth"),     exist_ok=True)
        os.makedirs(os.path.join(base_dir, cam + "_segmentation"), exist_ok=True)
    return


def load_canonical_mesh_from_asset(xml_path):
    txt = open(xml_path, 'r').read()
    wrapped = "<root>\n" + txt + "\n</root>"
    root = ET.fromstring(wrapped)

    # 2) find the first <mesh> anywhere under it
    mesh_elem = root.find(".//mesh")
    mesh_file = mesh_elem.attrib["file"]
    scale_str = mesh_elem.attrib.get("scale", "1 1 1")
    scales = [float(s) for s in scale_str.split()]

    # 3) resolve path and load+scale
    mesh_path = os.path.join(os.path.dirname(xml_path), mesh_file)
    mesh = trimesh.load(mesh_path, force="mesh")
    mesh.apply_scale(scales)
    return mesh

def save_json(path, data, nclass):
    if isinstance(data, dict) and "instruction" in data:
        with open(path, "w") as f:
            json.dump(data, f, indent=2)
        return
    
    os.makedirs(os.path.dirname(path), exist_ok=True)
    SEGMENTATION_METADATA = {}
    for cid in range(nclass):
        step = 255 // nclass if nclass > 1 else 0
        val = cid * step
        bgr = cv2.applyColorMap(
            np.array([[val]], dtype=np.uint8),
            cv2.COLORMAP_HSV
        )[0,0].tolist()
        SEGMENTATION_METADATA[cid] = {
            'id': cid,
            'color_bgr': [int(bgr[0]), int(bgr[1]), int(bgr[2])]
        }
    id_to_label = {
        0: 'box',
        1: 'gripper',
        2: 'robot',
        3: 'robot',
        4: 'object',
        5: 'background',
        6: 'table'
    }
    for cid, meta in SEGMENTATION_METADATA.items():
        meta['label'] = id_to_label.get(cid, 'none')

    # Combine metadata and data
    output = {
        'metadata': {
            'segmentation_classes': SEGMENTATION_METADATA
        },
        'data': data
    }
    with open(path, 'w') as f:
        json.dump(output, f, indent=2)

def writer_loop(q):
    """Background thread: write out images to disk."""
    for item in iter(q.get, None):
        path, arr = item
        folder = os.path.dirname(path)
        if not os.path.isdir(folder): 
            os.makedirs(folder, exist_ok=True)
        cv2.imwrite(path, arr)
    q.task_done()

def get_instruction(path, target_object):
    for obj, templates in INSTRUCTION_TEMPLATES.items():
        if obj in target_object.lower():
            return random.choice(templates)
    return "Perform the task as demonstrated."

def get_ee_pose(obs):
    """Return end-effector position & quaternion (x,y,z,w)."""
    pos  = obs["robot0_eef_pos"]
    quat = obs["robot0_eef_quat"]
    cor_quat = quat[[3, 0, 1, 2]]
    return np.array(pos), np.array(cor_quat)

def get_object_pose(obs, obj_name="Bread"):
    """Read the world pose of your object from the obs dict."""
    return np.array(obs[f"{obj_name}_pos"]), np.array(obs[f"{obj_name}_quat"])

def convert_depth_buffer_to_meters(depth_buffer, near=0.01, far=10.0):
    z_n = depth_buffer
    z_e = 2.0 * z_n - 1.0  # Convert [0,1] to [-1,1] (NDC space)
    depth = (2.0 * near * far) / (far + near - z_e * (far - near))
    return depth


def save_img_info(obs, base_dir, cam_names, step, action_vec, rew, done, robot, data_records):
    small_obs = {
        k: v for k, v in obs.items()
        if not (k.endswith("_image") or k.endswith("_depth") or k.endswith("_segmentation_class"))
    }
    # print("small_obs", small_obs.keys())
    # EEF
    eef_pos   = obs["robot0_eef_pos"]
    eef_quat = obs["robot0_eef_quat"]
    eef_quat = eef_quat[[3, 0, 1, 2]] 
    small_obs["state.pos_xyzquat_right"] = np.concatenate([eef_pos, eef_quat]).tolist()

    # joint state
    jidxs = robot._ref_joint_pos_indexes
    js    = robot.sim.data.qpos[jidxs].copy()
    small_obs["state.joint_state"] = obs["robot0_joint_pos"]
    #print(dir(robot.sim.data))

    
    # gripper opening
    gp    = obs["robot0_gripper_qpos"][0]
    #print("gripper position", gp)
    small_obs["state.position_normalized"] = [float(gp)]
    # print("small_obs", small_obs)

    # assemble record
    rec = {
        "step":        step,
        "observation": convert_obs(small_obs),
        "action":      action_vec.tolist(),
        "reward":      float(rew),
        "done":        bool(done),
    }

    known_objects = [obj.lower() for obj in INSTRUCTION_TEMPLATES.keys()]
    gripper_state = action_vec.tolist()[-1]
    for key, value in rec["observation"].items():
        if not key.endswith("_pos"):
            continue

        object_name = key.replace("_pos", "").lower()
        if object_name not in known_objects:
            continue

        if isinstance(value, list) and len(value) > 1:
            y_pos = value[1]
            if y_pos >= 0.05 and gripper_state <= -1.0:
                rec["done"] = True
                break

    delta = action_vec[0:3]    
    drot  = action_vec[3:6]    
    grip  = action_vec[6]

    rec["action.pos_xyzquat_right"]   = np.concatenate([delta, R.from_rotvec(drot).as_quat(scalar_first=True)]).tolist()
    rec["action.joint_state"]         = action_vec.tolist()
    rec["action.position_normalized"] = [grip]

    data_records.append(rec)

    for cam in cam_names:
        # RGB
        rgb = obs[f"{cam}_image"][..., ::-1]   # RGB→BGR
        rgb = np.flipud(rgb)
        p = os.path.join(base_dir, cam, f"{step:05d}.png")
        write_q.put((p, rgb))

        d = obs[f"{cam}_depth"].astype(np.float32)
        depth_meters = convert_depth_buffer_to_meters(d)  # from normalized depth
        depth_mm = (depth_meters * 1000).astype(np.uint16)
        depth_mm_flipped = np.flipud(depth_mm)

        depth_png_path = os.path.join(base_dir, f"{cam}_depth", f"{step:05d}.png")
        write_q.put((depth_png_path, depth_mm_flipped))
        # mn, mx = d.min(), d.max()
        # rng = mx - mn if (mx - mn) > 1e-6 else 1e-6
        # norm = ((d - mn) / rng * 255).astype(np.uint8)
        # norm = np.flipud(norm)
        # cam_depth = cam + "_depth"
        # p = os.path.join(base_dir, cam_depth, f"{step:05d}.png")
        # write_q.put((p, norm))

        # Segmentation
        mask = obs[f"{cam}_segmentation_class"]
        if mask.ndim == 3 and mask.shape[-1] == 1:
            mask = mask[:, :, 0]
        scaled  = (mask * (255 // (mask.max() + 1))).astype(np.uint8)
        colored = cv2.applyColorMap(scaled, cv2.COLORMAP_HSV)
        colored = np.flipud(colored)
        cam_seg = cam + "_segmentation"
        p = os.path.join(base_dir, cam_seg, f"{step:05d}.png")
        write_q.put((p, colored))
    
    return data_records


def move_xy_to_obj(env, robot, base_dir, cam_names, step, data_records):
    """
    Phase 1: Hold Z constant; move in X–Y only until within TOL_XY.
    """
    obs, rew, done, _ = env.step(robot.create_action_vector({"right": np.zeros(6), "right_gripper": np.array([-1.0])}))
    oscillate_cnt = total_steps = 0
    last_sign = None
    last_err  = None

    while True:
        
        rel_pos = np.array(obs["Bread_to_robot0_eef_pos"])  # [x,y,z] from bread→eef
        dx, dy, _ = rel_pos

        if abs(dx) < TOL_XY and abs(dy) < TOL_XY:
            break

        if last_sign is not None and sign != last_sign:
            oscillate_cnt += 1
        else:
            oscillate_cnt = 0
            total_steps  += 1
        
        if oscillate_cnt >= 6:
            break
        
        step_x = STEP_XY * np.sign(dx)
        step_y = STEP_XY * np.sign(dy)
        sign   = np.sign(step_x or step_y) 

        action = {
            "right":         np.array([ step_x, -step_y, 0,  0,0,0 ]),
            "right_gripper": np.array([-1.0]),
        }
        a = robot.create_action_vector(action)
        obs, rew, done,_ = env.step(a)
        data_records = save_img_info(obs, base_dir, cam_names, step, a, rew, done, robot, data_records)

        step += 1
        abort_if_too_many_steps(step)


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
            dz = np.array(obs["Bread_to_robot0_eef_pos"])[2]
        

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

        q_cur = obs["Bread_to_robot0_eef_quat"]
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

    while True:
        
        
        gx, gy = obs["robot0_eef_pos"][:2]
        dx, dy = gx - target_xy[0], gy - target_xy[1]
        # check tolerance
        # if abs(dx) < TOL_XY and abs(dy) < TOL_XY:
        if abs(dy) < TOL_XY and abs(dx) < TOL_XY: 
            break
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

def auto_pick_and_place(env, robot, write_q, base_dir, cam_names):
    """
    1) Move above object
    2) Descend & grasp
    3) Lift
    4) Move over target bin
    5) Descend & release
    """
   
    obs = env.reset()
    print(env.target_position)
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
    target_xy     = env.target_position
    # 1) move above object
    step, data_records = move_xy_to_obj(env, robot, base_dir, cam_names, step, data_records)
    if abort_if_too_many_steps(): return

    # 2) rotate to object 
    step, data_records = rotate_to(env, robot, base_dir, cam_names, step, data_records)
    if abort_if_too_many_steps(): return

    # 2b) align position again
    step, data_records = move_xy_to_obj(env, robot, base_dir, cam_names, step, data_records)
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
    target_object = "Bread"
    save_json(os.path.join(base_dir, "teleop_demo.json"), data_records, nclass)
    instr = get_instruction(base_dir, target_object)
    save_json(os.path.join(base_dir, "instruction.json"), {"instruction": instr}, nclass)

    print("Completed automatic pick-and-place")


if __name__ == "__main__":

    base_dir = f"/home/elisa/Documents/data/robosuite_automated/teleop_dataset_auto_{datetime.now():%Y%m%d_%H%M%S}"
    cam_names = ["left_side_view", "right_side_view",
                 "robot0_eye_in_hand_front", "robot0_eye_in_hand_back"]

    setup_dirs(base_dir, cam_names)



    write_q = queue.Queue()
    threading.Thread(target=writer_loop, args=(write_q,), daemon=True).start()

    ctrl_cfg = load_composite_controller_config(controller="BASIC")

    # 3) build env 
    env = BinToBinTransfer(
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
        initialization_noise=None
    )
    for cam in cam_names:
        save_intrinsic_extrinsic(env, cam, base_dir)

    robot = env.robots[0]
    
    xml_path = "/home/elisa/Documents/masterthesis/git/robosuite/robosuite/models/assets/objects/bread_asset.xml"
    bread_canonical = load_canonical_mesh_from_asset(xml_path)
    canonical_out = os.path.join(base_dir, "bread_canonical.ply")
    bread_canonical.export(canonical_out, file_type="ply", encoding="ascii")
    print(f"[+] wrote canonical bread mesh → {canonical_out}")

    auto_pick_and_place(env, robot, write_q, base_dir, cam_names)


    env.close()
    print(f"Data saved under {base_dir}")