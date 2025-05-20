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


STEP_SIZE = 0.3
UP_DOWN = 0.25

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

# key codes to delta‐vectors
KEY_TO_DELTA = {
    ord('w'): np.array([ STEP_SIZE,  0.0,  0.0]),  # (x+)
    ord('s'): np.array([-STEP_SIZE,  0.0,  0.0]),  # (x-)
    ord('a'): np.array([ 0.0,  STEP_SIZE,  0.0]),  # (y+)
    ord('d'): np.array([ 0.0, -STEP_SIZE,  0.0]),  #(y-)
    ord('u'): np.array([ 0.0,  0.0,  UP_DOWN]),    #  (z+)
    ord('j'): np.array([ 0.0,  0.0, -UP_DOWN]),    # (z-)
}
ROT_STEP = 0.4

KEY_TO_ROT = {
    ord('x'): np.array([ROT_STEP, 0, 0]),   # rotate x+
    ord('c'): np.array([-ROT_STEP, 0, 0]),  # rotate x-
    ord('l'): np.array([0, ROT_STEP, 0]),   # rotate y+
    ord('k'): np.array([0, -ROT_STEP, 0]),  # rotate y-
    ord('n'): np.array([0, 0, ROT_STEP]),   # rotate z+
    ord('m'): np.array([0, 0, -ROT_STEP]),  # rotate z-
}

TARGET_EEF_POS = np.array([0.58232898, 0.21374725, 0.18609748])

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


# def save_camera_info(env, cam, base_dir):
#     cam_id = env.sim.model.camera_name2id(cam)
#     # Intrinsics
#     fovy = env.sim.model.cam_fovy[cam_id]
#     width = env.camera_widths[env.camera_names.index(cam)]
#     height = env.camera_heights[env.camera_names.index(cam)]
#     f = 0.5 * height / np.tan(np.deg2rad(fovy / 2))
#     K = [[f, 0, width/2], [0, f, height/2], [0,0,1]]
#     intr = {"fovy_deg": fovy, "fx": f, "fy": f, "cx": width/2, "cy": height/2, "K": K}
#     # Extrinsics
#     pos = env.sim.model.cam_pos[cam_id].tolist()
#     quat = env.sim.model.cam_quat[cam_id]
#     Rm = R.from_quat([quat[1], quat[2], quat[3], quat[0]]).as_matrix().tolist()
#     ext = {"pos": pos, "quat_wxyz": quat.tolist(), "R": Rm}
#     info = {"camera": cam, "intrinsics": intr, "extrinsics": ext}
#     with open(os.path.join(base_dir, f"{cam}_camera_info.json"), 'w') as f:
#         json.dump(info, f, indent=2)


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
    for item in iter(q.get, None):
        path, arr = item
        cv2.imwrite(path, arr)
    q.task_done()

def get_instruction(path):
    for obj, templates in INSTRUCTION_TEMPLATES.items():
        if obj in path.lower():
            return random.choice(templates)
    return "Perform the task as demonstrated."


def main(task): 
    #task = "robotic_cell"
    base_dir = "../teleop_dataset_eef/teleop_dataset_" + str(task) + "_bread_" + datetime.now().strftime("%Y%m%d_%H%M%S")
    print(base_dir)
    os.makedirs(base_dir, exist_ok=True)
    
    ctrl_cfg = load_composite_controller_config(controller="BASIC")
    if task == "blue_bin_multi_object_picking":
        EnvClass = BinToBinTransfer
        cam_names = ["left_side_view", "right_side_view", "robot0_eye_in_hand_front", "robot0_eye_in_hand_back"]
    elif task == "robotic_cell":
        EnvClass = PickPlace
        cam_names = ["robot0_robotview", "frontview"]
    else:
        EnvClass = Lift
        cam_names = ["frontview", "agentview", "sideview"]

    setup_dirs(base_dir, cam_names)
    env = EnvClass(
        robots="Panda",
        controller_configs=ctrl_cfg,
        has_renderer=False,
        has_offscreen_renderer=True,
        use_camera_obs=True,
        camera_names=cam_names,
        camera_heights=[480]*len(cam_names),
        camera_widths=[640]*len(cam_names),
        camera_depths=True,
        camera_segmentations=["class"]*len(cam_names),
        control_freq=10,
        ignore_done=True,
        hard_reset=False,
    )
    obs = env.reset()
    xml_path = "/home/elisa/Documents/masterthesis/git/robosuite/robosuite/models/assets/objects/bread_asset.xml"
    bread_canonical = load_canonical_mesh_from_asset(xml_path)
    canonical_out = os.path.join(base_dir, "bread_canonical.ply")
    bread_canonical.export(canonical_out, file_type="ply", encoding="ascii")
    print(f"[+] wrote canonical bread mesh → {canonical_out}")


    first_cam_seg = f"{cam_names[0]}_segmentation_class"
    seg = obs.get(first_cam_seg)
    if seg is not None:
        if seg.ndim == 3 and seg.shape[-1] == 1:
            seg = seg[:, :, 0]
        nclass = int(seg.max() + 1)
    else:
        nclass = 1
    print(f"Detected {nclass} segmentation classes (IDs 0..{nclass-1})")

    # Save camera infos
    for cam in cam_names:
        save_intrinsic_extrinsic(env, cam, base_dir)

    write_q = queue.Queue()
    threading.Thread(target=writer_loop, args=(write_q,), daemon=True).start()

    robot = env.robots[0]
    zero_act = robot.create_action_vector({"right": np.zeros(6), "right_gripper": np.zeros(1)})
    env.reset()
    env.step(zero_act)

    cv2.namedWindow("teleop", cv2.WINDOW_NORMAL)
    step = 0
    data_records = []
    grip = 0.0

    while True:

        zero_action = robot.create_action_vector({"right": np.zeros(6), "right_gripper": np.array([0.0])})
        obs, _, _, _ = env.step(zero_action)


        stacked_imgs = []
        for idx in (0,2):
            cam_name = cam_names[idx]
            if f"{cam_name}_image" in obs:
                img = cv2.cvtColor(obs[f"{cam_name}_image"], cv2.COLOR_RGB2BGR)
                img = cv2.flip(img, 0)
                img = cv2.resize(img, (320, 240))
                cv2.putText(img, cam_name, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 0), 2)
                stacked_imgs.append(img)

        rows = [np.hstack(stacked_imgs[i:i+2]) for i in range(0, len(stacked_imgs), 2)]
        stacked = np.vstack(rows)
        cv2.imshow("teleop", stacked)

        key = cv2.waitKey(10) & 0xFF
        if key in (13, 10):  # ENTER
            print("Starting teleoperation and recording...")
            break
        elif key in (ord('q'), ord('Q'), 27):  # quit
            print("Exiting before recording.")
            env.close()
            cv2.destroyAllWindows()
            return

    try:
        while True:
            obs = env.sim.render(640, 480, camera_name=cam_names[0])
            key = cv2.waitKey(1) & 0xFF
            if key in (ord('q'), ord('Q'), 27):
                break
            delta = KEY_TO_DELTA.get(key, np.zeros(3))
            drot = KEY_TO_ROT.get(key, np.zeros(3))
            # update gripper
            if key in (ord('o'), ord('O')): grip=-1.0
            elif key in (ord('p'), ord('P')): grip=1.0
            else: grip=grip

            action = robot.create_action_vector({
                "right": np.concatenate([delta, drot]),
                "right_gripper": np.array([grip])
            })
            obs, rew, done, info = env.step(action)

            eef_pos     = robot._hand_pos["right"]               # [x,y,z]
            eef_quat    = robot._hand_quat["right"]              # [x,y,z,w]
            joint_state = robot.sim.data.qpos[robot._ref_joint_pos_indexes].copy()  # 7‐joint vector

            # gripper qpos mean as a scalar “position_m”
            gripper_idxs = list(robot._ref_gripper_joint_pos_indexes.values())
            gripper_qpos = robot.sim.data.qpos[gripper_idxs]
            gripper_pos_m= np.array([gripper_qpos.mean()]) 

            small_obs = {k: v for k,v in obs.items()
                            if not k.endswith("_image") and not k.endswith("_depth") and not k.endswith("_segmentation_class")}    

            rec = {"step": step, "observation": convert_obs(small_obs), "action": action.tolist(), "reward": float(rew), "done": bool(done)}

            rec["observation"]["state.pos_xyzquat_right"] = np.concatenate([eef_pos, eef_quat]).tolist()
            rec["observation"]["state.joint_state"]      = joint_state.tolist()
            rec["observation"]["state.position_normalized"]       = gripper_pos_m.tolist()

            rec["action.pos_xyzquat_right"] = np.concatenate([delta, R.from_rotvec(drot).as_quat()]).tolist()
            rec["action.joint_state"]      = action[:7].tolist()
            rec["action.position_normalized"]       = [grip]
            
            for cam in cam_names:
                for mod in ["image","depth","segmentation"]:
                    if mod == "image": 
                        field = f"{cam}_file"
                        path  = os.path.join(base_dir, cam, mod, f"{step:05d}.png")
                    else: 
                        field = f"{cam}_{mod}_file"
                        path  = os.path.join(base_dir, cam, mod, f"{step:05d}.png")
                    rec[field] = path

            # "done” logic:
            known_objects = [obj.lower() for obj in INSTRUCTION_TEMPLATES.keys()]
            for key, value in rec["observation"].items():
                if not key.endswith("_pos"):
                    continue

                object_name = key.replace("_pos", "").lower()
                if object_name not in known_objects:
                    continue

                if isinstance(value, list) and len(value) > 1:
                    y_pos = value[1]
                    if y_pos >= 0.05 and grip <= -1.0:
                        rec["done"] = True
                        break


            data_records.append(rec)
            display_cams = env.camera_names[:4] if len(env.camera_names) >= 4 else env.camera_names[:2]
            for cam in cam_names:

                rgb_key = f"{cam}_image"
                if rgb_key in obs:
                    img = cv2.cvtColor(obs[rgb_key], cv2.COLOR_RGB2BGR)
                    img = cv2.flip(img, 0)
                    path = os.path.join(base_dir, f"{cam}", f"{step:05d}.png")
                    write_q.put((path, img))

                dep_key = f"{cam}_depth"
                if dep_key in obs:
                    d = obs[dep_key]
                    mn, mx = d.min(), d.max()
                    norm = ((d - mn) / (mx - mn + 1e-6) * 255).astype(np.uint8)
                    norm = cv2.flip(norm, 0)
                    path = os.path.join(base_dir, f"{cam}_depth", f"{step:05d}.png")
                    write_q.put((path, norm))

                seg_key = f"{cam}_segmentation_class"
                if seg_key in obs:
                    mask = obs[seg_key]
                    if mask.ndim == 3 and mask.shape[-1] == 1:
                        mask = mask[:, :, 0]
                    mask = mask.astype(np.uint8)
                    scaled = (mask * (255 // max(nclass, 1))).astype(np.uint8)
                    colored = cv2.applyColorMap(scaled, cv2.COLORMAP_HSV)
                    colored = cv2.flip(colored, 0)
                    path = os.path.join(base_dir, f"{cam}_segmentation", f"{step:05d}.png")
                    write_q.put((path, colored))

            # display rotated teleop overview
            stacked_imgs = []
            for idx in (0,2):
                cam = display_cams[idx]
                rgb_key = f"{cam}_image"
                if rgb_key in obs:
                    img = cv2.cvtColor(obs[rgb_key], cv2.COLOR_RGB2BGR)
                    img = cv2.flip(img, 0)
                    img = cv2.resize(img, (320,240))
                    cv2.putText(img, cam, (10,30),
                                cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255,255,0), 2)
                    stacked_imgs.append(img)
            rows   = [np.hstack(stacked_imgs[i:i+2]) for i in range(0, len(stacked_imgs), 2)]
            stacked= np.vstack(rows)
            
            cv2.imshow("teleop", stacked)

            step += 1
            if done:
                break

    finally:
        write_q.put(None)
        env.close()
        cv2.destroyAllWindows()
        # save teleop log & instruction
        save_json(os.path.join(base_dir, "teleop_demo.json"), data_records, nclass)
        instr = get_instruction(base_dir)
        save_json(os.path.join(base_dir, "instruction.json"), {"instruction": instr}, nclass)
        print(f"Saved {step} steps, data to {base_dir}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--task", type=str, default="blue_bin_multi_object_picking")
    args = parser.parse_args()
    main(args.task)