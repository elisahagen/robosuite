import numpy as np
import robosuite as suite
import os
import cv2
import argparse
import threading
import queue
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
import trimesh
from robosuite.models.objects import BreadObject
import xml.etree.ElementTree as ET

INSTRUCTION_TEMPLATES = {
    "milk": [
        "Please grab the milk and place it on the table.",
        "Pick up the milk carton and move it to the tray.",
        "Lift the milk and drop it in the basket.",
        "Take the milk and place it near the sink."
    ],
    "bread": [
        "Pick up the bread and put it in the basket.",
        "Grab the loaf of bread and move it to the counter.",
        "Lift the bread and drop it on the tray.",
        "Please relocate the bread to the plate."
    ],
    "can": [
        "Take the can and place it on the shelf.",
        "Pick up the can and move it to the bin.",
        "Grab the can and drop it near the box.",
        "Relocate the can to the storage area."
    ],
    "juice": [
        "Lift the juice bottle and set it on the table.",
        "Grab the juice container and drop it in the tray.",
        "Take the juice and place it on the counter."
    ],
    "cereals": [
        "Lift the cereal box and set it on the table.",
        "Grab the cereal box and drop it in the tray.",
        "Take the cereal box and place it on the counter."
    ],
    "bottle": [
        "Lift the bottle and set it on the table.",
        "Grab the bottle and drop it in the tray.",
        "Take the bottle and place it on the counter."
    ],
    "cube": [
        "Lift the cube and set it on the table.",
        "Grab the cube and drop it in the tray.",
        "Take the cube and place it on the counter."
    ],
    "box": [
        "Lift the box and set it on the table.",
        "Grab the box and drop it in the tray.",
        "Take the box and place it on the counter."
    ],
    "capsule": [
        "Lift the capsule and set it on the table.",
        "Grab the capsule and drop it in the tray.",
        "Take the capsule and place it on the counter."
    ],
    "cylinder": [
        "Lift the cylinder and set it on the table.",
        "Grab the cylinder and drop it in the tray.",
        "Take the cylinder and place it on the counter."
    ]
}

STEP_SIZE = 0.3
UP_DOWN   = 0.25
TOL_XY    = 0.2
TOL_Z     = 0.01
HOME_JOINTS = [0., np.pi/4., 0., -np.pi/4., 0., np.pi/2., 0.]

STEP_XY   = 0.3
STEP_Z    = 0.25
STEP_ROT  = 0.2  # rad per step (~23°)
STEP_TIL = 0.2  # rad per step (~23°)
TOL_XY    = 0.01
TOL_Z     = 0.01
TOL_YAW   = 0.01 

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
    focal_length = 0.5 * height / np.tan(np.deg2rad(fovy / 2))

    intrinsics = {
        "camera_name": cam,
        "image_width": width,
        "image_height": height,
        "fovy_deg": fovy,
        "focal_length_px": focal_length,
        "intrinsic_matrix_K": [
            [focal_length, 0, width / 2],
            [0, focal_length, height / 2],
            [0, 0, 1]
        ]
    }

    # --- Extrinsics ---
    cam_pos = env.sim.model.cam_pos[cam_id].tolist()
    cam_quat = env.sim.model.cam_quat[cam_id]  # [w, x, y, z]
    rotation_matrix = R.from_quat([cam_quat[1], cam_quat[2], cam_quat[3], cam_quat[0]]).as_matrix().tolist()

    extrinsics = {
        "position_xyz": cam_pos,
        "rotation_quaternion_wxyz": cam_quat.tolist(),
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

def get_instruction_from_path(path):
    for obj_name in INSTRUCTION_TEMPLATES:
        if obj_name in path.lower():
            return random.choice(INSTRUCTION_TEMPLATES[obj_name])
    return "Perform the task with the object as demonstrated."
    

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
        os.makedirs(os.path.dirname(path), exist_ok=True)
        cv2.imwrite(path, arr)
    q.task_done()

def get_instruction(path):
    for obj, templates in INSTRUCTION_TEMPLATES.items():
        if obj in path.lower():
            return random.choice(templates)
    return "Perform the task as demonstrated."

def get_ee_pose(obs):
    """Return end-effector position & quaternion (x,y,z,w)."""
    pos  = obs["robot0_eef_pos"]
    quat = obs["robot0_eef_quat"]
    return np.array(pos), np.array(quat)

def get_object_pose(obs, obj_name="Bread"):
    """Read the world pose of your object from the obs dict."""
    return np.array(obs[f"{obj_name}_pos"]), np.array(obs[f"{obj_name}_quat"])

def save_img_info(obs, base_dir, cam_names, step, action_vec, rew, done, robot, data_records):
    # strip out image/depth/seg keys
    small_obs = {
        k: v for k, v in obs.items()
        if not (k.endswith("_image") or k.endswith("_depth") or k.endswith("_segmentation_class"))
    }
    # EEF
    eef_pos   = robot._hand_pos["right"]
    eef_quat  = robot._hand_quat["right"]
    small_obs["state.pos_xyzquat_right"] = np.concatenate([eef_pos, eef_quat]).tolist()

    # joint state
    jidxs = robot._ref_joint_pos_indexes
    js    = robot.sim.data.qpos[jidxs].copy()
    small_obs["state.joint_state"] = js.tolist()

    # gripper opening
    gidxs = list(robot._ref_gripper_joint_pos_indexes.values())
    gp    = robot.sim.data.qpos[gidxs].mean()
    small_obs["state.position_normalized"] = [float(gp)]

    # assemble record
    rec = {
        "step":        step,
        "observation": convert_obs(small_obs),
        "action":      action_vec.tolist(),
        "reward":      float(rew),
        "done":        bool(done),
    }

    delta = action_vec[0:3]    
    drot  = action_vec[3:6]    
    grip  = action_vec[6]

    rec["action.pos_xyzquat_right"]   = np.concatenate([delta, R.from_rotvec(drot).as_quat()]).tolist()
    rec["action.joint_state"]         = action_vec.tolist()
    rec["action.position_normalized"] = [grip]

    data_records.append(rec)

    for cam in cam_names:
        # RGB
        rgb = obs[f"{cam}_image"][..., ::-1]   # RGB→BGR
        rgb = np.flipud(rgb)
        p = os.path.join(base_dir, cam, "image", f"{step:05d}.png")
        write_q.put((p, rgb))

        # Depth
        d = obs[f"{cam}_depth"].astype(np.float32)
        mn, mx = d.min(), d.max()
        # was d.ptp(), now:
        rng = mx - mn if (mx - mn) > 1e-6 else 1e-6
        norm = ((d - mn) / rng * 255).astype(np.uint8)
        norm = np.flipud(norm)
        p = os.path.join(base_dir, cam, "depth", f"{step:05d}.png")
        write_q.put((p, norm))

        # Segmentation
        mask = obs[f"{cam}_segmentation_class"]
        if mask.ndim == 3 and mask.shape[-1] == 1:
            mask = mask[:, :, 0]
        scaled  = (mask * (255 // (mask.max() + 1))).astype(np.uint8)
        colored = cv2.applyColorMap(scaled, cv2.COLORMAP_HSV)
        colored = np.flipud(colored)
        p = os.path.join(base_dir, cam, "segmentation", f"{step:05d}.png")
        write_q.put((p, colored))
    
    return data_records


def move_xy_to_obj(env, robot, base_dir, cam_names, step, data_records):
    """
    Phase 1: Hold Z constant; move in X–Y only until within TOL_XY.
    """
    obs, rew, done, _ = env.step(robot.create_action_vector({"right": np.zeros(6), "right_gripper": np.array([-1.0])}))
    while True:
        
        rel_pos = np.array(obs["Bread_to_robot0_eef_pos"])  # [x,y,z] from bread→eef
        dx, dy, _ = rel_pos

        if abs(dx) < TOL_XY and abs(dy) < TOL_XY:
            print("break")
            break

        step_x = STEP_XY * np.sign(dx)
        step_y = STEP_XY * np.sign(dy)

        action = {
            "right":         np.array([ step_x, -step_y, 0,  0,0,0 ]),
            "right_gripper": np.array([-1.0]),
        }
        a = robot.create_action_vector(action)
        obs, rew, done,_ = env.step(a)
        data_records = save_img_info(obs, base_dir, cam_names, step, a, rew, done, robot, data_records)

        step += 1


    return step, data_records


def move_z_to(env, robot, base_dir, cam_names, step, target, data_records):
    """
    Phase 2: With X–Y already aligned, move just in Z until within TOL_Z.
    """
    if target is not None:
        obs, rew, done, _ = env.step(robot.create_action_vector({"right": np.zeros(6), "right_gripper": np.array([+1.0])}))  
    else: 
        obs, rew, done, _ = env.step(robot.create_action_vector({"right": np.zeros(6), "right_gripper": np.array([-1.0])}))  
    print("in z")

    while True:
        
        if target is not None: 
            dz = target[2] - np.array(obs["robot0_eef_pos"])[2]
        else:
            dz = np.array(obs["Bread_to_robot0_eef_pos"])[2]
        

        if abs(dz) < TOL_Z:
            print("break z")
            break

        step_z = STEP_Z * np.sign(dz)
        if target is not None:
            for i in range(20): 
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
        
    return step, data_records 

def rotate_to(env, robot, base_dir, cam_names, step, data_records):
    """
    Rotate your end‐effector so that its axes line up with the object,
    stepping in ROT_STEP until the axis‐angle error is below ROT_TOL.
    """
    obs, rew, done, _ = env.step(robot.create_action_vector({
        "right":         np.zeros(6),
        "right_gripper": np.array([-1.0])
    }))
    

    q = obs["Bread_to_robot0_eef_quat"]  
    R_obj2eef = R.from_quat([q[1], q[2], q[3], q[0]])
    yaw_err, _, _ = R_obj2eef.inv().as_euler("zyx", degrees=False)

    while abs(yaw_err) > STEP_TIL:
        step_ang  = np.sign(yaw_err) * STEP_ROT
        action = {
            "right":         np.array([0.0, 0.0, 0.0, 0.0, 0.0, step_ang]),
            "right_gripper": np.array([-1.0]),  
        }
        a = robot.create_action_vector(action)
        obs, rew, done,_ = env.step(a)


        for cam in cam_names:
            img = obs[f"{cam}_image"][..., ::-1]   # RGB→BGR
            img = np.flipud(img)
            path = os.path.join(base_dir, cam, "image", f"{step:05d}.png")
            write_q.put((path, img))
        step += 1
        data_records = save_img_info(obs, base_dir, cam_names, step, a, rew, done, robot, data_records)

        q = obs["Bread_to_robot0_eef_quat"]
        R_obj2eef = R.from_quat([q[1], q[2], q[3], q[0]])
        yaw_err, _, _ = R_obj2eef.inv().as_euler("zyx", degrees=False)

    return step, data_records 

def move_xy_to_target(env, robot, base_dir, cam_names, step, target_xy, data_records): 

    obs, rew, done, _ = env.step(robot.create_action_vector({"right": np.zeros(6), "right_gripper": np.array([+1.0])}))

    while True:
        
        
        gx, gy = obs["robot0_eef_pos"][:2]
        dx, dy = gx - target_xy[0], gy - target_xy[1]
        # check tolerance
        if abs(dx) < TOL_XY and abs(dy) < TOL_XY:
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

    for _ in range(15):
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
    a = robot.create_action_vector(action)
    obs, rew, done, _ = env.step(a)
    data_records = save_img_info(obs, base_dir, cam_names, step, a, rew, done, robot, data_records)
    step += 1

    for _ in range(15):
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
    step = 0
    target = None
    data_records = []

    ee_pos, _     = get_ee_pose(obs)
    target_xy     = [0.13, 0.13, 0.6]
    # 1) move above object
    step, data_records = move_xy_to_obj(env, robot, base_dir, cam_names, step, data_records)

    # 2) rotate to object 
    step, data_records = rotate_to(env, robot, base_dir, cam_names, step, data_records)

    # 3) descend to object
    step, data_records = move_z_to(env, robot, base_dir, cam_names, step, target, data_records)

    # 4) close gripper
    obs,rew, done,  _ = env.step(robot.create_action_vector({"right": np.zeros(6), "right_gripper": np.array([+1.0])}))
    step += 1

    # 5) lift up 15cm
    target =  np.array([0,0,0.13])
    step, data_records = move_z_to(env, robot, base_dir, cam_names, step, target, data_records)

    # 6) move over bin at X=0.3, Y=0
    step, data_records = move_xy_to_target(env, robot, base_dir, cam_names, step, target_xy, data_records)

    nclass = 6
    save_json(os.path.join(base_dir, "teleop_demo.json"), data_records, nclass)
    instr = get_instruction(base_dir)
    save_json(os.path.join(base_dir, "instruction.json"), {"instruction": instr}, nclass)

    print("✅ Completed automatic pick-and-place")


if __name__ == "__main__":

    base_dir = f"../teleop_dataset_auto_{datetime.now():%Y%m%d_%H%M%S}"
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
        camera_heights=[480]*4,
        camera_widths =[640]*4,
        camera_depths=True,
        camera_segmentations=["class"]*4,
        control_freq=20,
        ignore_done=True,
        hard_reset=False,
    )
    for cam in cam_names:
        save_intrinsic_extrinsic(env, cam, base_dir)

    robot = env.robots[0]

    # 4) run auto pick‐and‐place
    auto_pick_and_place(env, robot, write_q, base_dir, cam_names)

    env.close()
    print(f"Data saved under {base_dir}")