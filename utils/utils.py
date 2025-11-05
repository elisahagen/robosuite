import numpy as np
import os 
import json
from scipy.spatial.transform import Rotation as R
import cv2
import threading
import queue
from robosuite.utils import transform_utils as T
from instruction_templates import InstructionTemplatesLevel1, InstructionTemplatesLevel2, InstructionTemplatesLevel3
import trimesh
import xml.etree.ElementTree as ET
from robosuite.utils.camera_utils import (
    get_camera_intrinsic_matrix
)

def writer_loop(q):
    """Background thread: write out images to disk."""
    for item in iter(q.get, None):
        path, arr = item
        folder = os.path.dirname(path)
        if not os.path.isdir(folder): 
            os.makedirs(folder, exist_ok=True)
        cv2.imwrite(path, arr)
    q.task_done()


write_q = queue.Queue()
threading.Thread(target=writer_loop, args=(write_q,), daemon=True).start()

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

    # Handle list of steps
    base, ext = os.path.splitext(path)
    output = {
        "data": [],
        "metadata": {
            "segmentation": SEGMENTATION_METADATA
        }
    }
    for i, step in enumerate(data):
        step_data = convert_obs(step)
  
        output["data"].append(step_data)

    with open(base, 'w') as f:
        json.dump(output, f, indent=2)


def convert_obs(obs):
    """
    Convert observations to JSON‐serializable format.
    """
    if isinstance(obs, np.ndarray):
        return obs.tolist()
    elif isinstance(obs, dict):
        return {
            str(key): convert_obs(val) for key, val in obs.items()
            if not isinstance(val, (type(convert_obs), type(convert_obs)))
        }
    elif isinstance(obs, list):
        return [convert_obs(elem) for elem in obs if not isinstance(elem, (type(convert_obs), type(convert_obs)))]
    elif isinstance(obs, (int, float, str, bool)):
        return obs
    else:
        return str(obs) 
    
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

    # print(f"Saved camera info for {cam} to {save_path}")

def convert_depth_buffer_to_meters(depth_buffer, near=0.01, far=10.0):
    z_n = depth_buffer
    z_e = 2.0 * z_n - 1.0  # Convert [0,1] to [-1,1] (NDC space)
    depth = (2.0 * near * far) / (far + near - z_e * (far - near))
    return depth

def save_img_info(obs, env, base_dir, cam_names, step, action_vec, rew, done, robot, data_records = None):
    small_obs = {
        k: v for k, v in obs.items()
        if not (k.endswith("_image") or k.endswith("_depth") or k.endswith("_segmentation_class"))
    }
    # print("small_obs", small_obs.keys())
    # EEF
    # print(obs.keys())
    eef_pos   = obs["robot0_eef_pos"]
    eef_quat = obs["robot0_eef_quat"]
    # eef_quat = eef_quat[[3, 0, 1, 2]] 
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
    delta_pos = action_vec[:3]
    delta_axisangle = action_vec[3:6]
    print("actionshape", action_vec.shape)
    gripper = action_vec[-1]

    # Convert relative rotation to quaternion
    delta_quat = T.axisangle2quat(delta_axisangle)

    # Combine rotations: q_next = delta * current
    target_quat = T.quat_multiply(delta_quat, eef_quat)

    # Compute next absolute position
    target_pos = eef_pos + delta_pos

    # Convert quaternion → matrix → Euler angles
    target_rotmat = T.quat2mat(target_quat)
    target_euler = T.mat2euler(target_rotmat)

    # Combine into absolute action (pos + rotation)
    action_abs = np.concatenate([target_pos, target_euler, [gripper]])

    # assemble record
    rec = {
        "step":        step,
        "observation": convert_obs(small_obs),
        "action":      action_vec.tolist(),
        "action_abs":  action_abs.tolist(),
        "reward":      float(rew),
        "done":        bool(done),
    }

    
    known_objects = set()
    

    sim = env.sim

    # Get IDs of your fingertip-like sites
    left_id = sim.model.site_name2id("gripper0_right_ee_y")
    right_id = sim.model.site_name2id("gripper0_right_ee_x")

    # Get their 3D world positions
    left_pos = sim.data.site_xpos[left_id].copy()
    right_pos = sim.data.site_xpos[right_id].copy()

    # print("Left keypoint:", left_pos)
    # print("Right keypoint:", right_pos)

    # Append to your data record (raw 3D)
    data_records.append({
        "step": step,
        "left_finger_pos": left_pos.tolist(),
        "right_finger_pos": right_pos.tolist(),
    })

    # ---- Store everything ----


    # Add all object keys from each template level
    for tmpl in (InstructionTemplatesLevel1, InstructionTemplatesLevel2, InstructionTemplatesLevel3):
        known_objects.update(obj.lower() for obj in tmpl.keys())

    known_objects = list(known_objects)

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
        
        
        raw_labels = np.flipud(mask.astype(np.uint16))
        cam_seg_raw = f"{cam}_segmentation_class"
        p_raw = os.path.join(base_dir, cam_seg_raw, f"{step:05d}.png")
        write_q.put((p_raw, raw_labels))

        scaled  = (mask * (255 // (mask.max() + 1))).astype(np.uint8)
        colored = cv2.applyColorMap(scaled, cv2.COLORMAP_HSV)
        colored = np.flipud(colored)
        cam_seg = cam + "_segmentation"
        p = os.path.join(base_dir, cam_seg, f"{step:05d}.png")
        write_q.put((p, colored))
    
    return data_records
