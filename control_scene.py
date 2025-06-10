import numpy as np
import robosuite as suite
import os
import cv2
import argparse
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

KEY_TO_ROTATION = {
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
    
def main(task): 
    #task = "robotic_cell"
    base_dir = "../teleop_dataset_eef/teleop_dataset_" + str(task) + "_bread_" + datetime.now().strftime("%Y%m%d_%H%M%S")
    print(base_dir)
    os.makedirs(base_dir, exist_ok=True)
    
    
    controller_config = load_composite_controller_config(controller="BASIC")
    if task == "blue_bin_multi_object_picking":
        camera_names = ["left_side_view", "right_side_view", "robot0_eye_in_hand_front", "robot0_eye_in_hand_back"]
        for cam in camera_names:
            os.makedirs(os.path.join(base_dir,cam),     exist_ok=True)
            os.makedirs(os.path.join(base_dir,cam+"_depth"), exist_ok=True)
    
        env = BinToBinTransfer(
            robots="Panda",
            controller_configs=controller_config,
            has_renderer=False,           
            has_offscreen_renderer=True,
            use_camera_obs=True,
            render_camera="frontview",
            camera_names=camera_names,
            camera_heights=480,
            camera_widths=640,
            camera_depths=True,         
            camera_segmentations=["class", "class", "class", "class"],
            control_freq=10,
            ignore_done=True,
            hard_reset=False
        )
    if task == "robotic_cell":
        camera_names = ['robot0_robotview', "frontview"]
        for cam in camera_names:
            os.makedirs(os.path.join(base_dir,cam),     exist_ok=True)
            os.makedirs(os.path.join(base_dir,cam+"_depth"), exist_ok=True)
        env = PickPlace(
            robots="Panda",
            controller_configs=controller_config,
            has_renderer=False,           
            has_offscreen_renderer=True,
            use_camera_obs=True,
            render_camera="robot0_robotview",
            camera_names=camera_names,
            camera_heights=480,
            camera_widths=640,
            camera_depths=True,
            control_freq=10,
            ignore_done=True,
            hard_reset=False
        )
    elif task == "lift":
        camera_names = ["frontview", "agentview", "sideview"]
        for cam in camera_names:
            os.makedirs(os.path.join(base_dir,cam),     exist_ok=True)
            os.makedirs(os.path.join(base_dir,cam+"_depth"), exist_ok=True)

        env = Lift(
            robots="Panda",
            controller_configs=controller_config,
            has_renderer=False,            
            has_offscreen_renderer=True,
            use_camera_obs=True,
            render_camera="frontview",
            camera_names=camera_names,
            camera_heights=240,
            camera_widths=320,
            camera_depths=True,
            control_freq= 10,
            ignore_done=True,
        )

    obs = env.reset()

    first_cam_seg = f"{camera_names[0]}_segmentation_class"
    seg = obs.get(first_cam_seg)
    if seg is not None:
        if seg.ndim == 3 and seg.shape[-1] == 1:
            seg = seg[:, :, 0]
        nclass = int(seg.max() + 1)
    else:
        nclass = 1
    print(f"Detected {nclass} segmentation classes (IDs 0..{nclass-1})")

    for cam in env.camera_names:
        save_intrinsic_extrinsic(env, cam, base_dir)

    robot = env.robots[0]
    done = False

    step = 0

    gripper_state = 0.0

    data_log = []

    cv2.namedWindow("teleop", cv2.WINDOW_NORMAL)
    print("Controls:")
    print("  W/S/A/D: move in x/y plane")
    print("  X/C:     rotate in x+/i plane")
    print("  L/K:     rotate in y+/y- plane")
    print("  N/M:     rotate in z+/z- plane")
    print("  U/J:     move up/down")
    print("  O:       open gripper")
    print("  P:       close gripper")
    print("  V:       toggle camera view")
    print("  Q:       quit")

    display_cams = env.camera_names[:4] if len(env.camera_names) >= 4 else env.camera_names[:2]

    while True:

        zero_action = robot.create_action_vector({"right": np.zeros(6), "right_gripper": np.array([0.0])})
        obs, _, _, _ = env.step(zero_action)


        stacked_imgs = []
        for cam_name in display_cams:
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
            zero_action = robot.create_action_vector({"right": np.zeros(6), "right_gripper": np.array([0.0])})
            obs, _, _, _ = env.step(zero_action)

            # front_img = cv2.cvtColor(obs["frontview_image"], cv2.COLOR_RGB2BGR)
            # gripper_img = cv2.cvtColor(obs["robot0_eye_in_hand_image"], cv2.COLOR_RGB2BGR)
            # agent_img = cv2.cvtColor(obs["agentview_image"], cv2.COLOR_RGB2BGR)
            # bird_img = cv2.cvtColor(obs["birdview_image"], cv2.COLOR_RGB2BGR)
            # gripper_img = cv2.flip(gripper_img, 180)
            # front_img = cv2.flip(front_img, 0)
            # agent_img = cv2.flip(agent_img, 0)
            # bird_img = cv2.flip(bird_img, 0)
            # cv2.putText(front_img, "Front View", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0,255,0), 2)
            # cv2.putText(gripper_img, "Gripper View", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0,255,0), 2)
            # cv2.putText(agent_img, "Agent View", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0,255,0), 2)
            # cv2.putText(bird_img, "Bird View", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0,255,0), 2)
            # top_row = np.hstack((front_img, gripper_img))
            # bottom_row = np.hstack((agent_img, bird_img))
            # stacked = np.vstack((top_row, bottom_row))
            # cv2.imshow("teleop", stacked)

            stacked_imgs = []
            for cam_name in display_cams:
                img_key = f"{cam_name}_image"
                if img_key in obs:
                    img = cv2.cvtColor(obs[img_key], cv2.COLOR_RGB2BGR)
                    img = cv2.flip(img, 0)
                    img = cv2.resize(img, (320, 240))
                    cv2.putText(img, cam_name, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 0), 2)
                    stacked_imgs.append(img)

            rows = [np.hstack(stacked_imgs[i:i+2]) for i in range(0, len(stacked_imgs), 2)]
            stacked = np.vstack(rows)

            cv2.imshow("teleop", stacked)


            delta = np.zeros(3)
            delta_rot = np.zeros(3)
            key = cv2.waitKey(10) & 0xFF

            # Movement
            if key in KEY_TO_DELTA:
                delta += KEY_TO_DELTA[key]
            # Rotation
            if key in KEY_TO_ROTATION:
                delta_rot += KEY_TO_ROTATION[key]
            # Gripper
            if key == ord('p'):
                gripper_state = 1.0
                print("Gripper: CLOSED")
            elif key == ord('o'):
                gripper_state = -1.0
                print("Gripper: OPEN")
            elif key in (ord('q'), ord('Q'), 27):
                print("Quitting.")
                break
            
            
            # Compose and send action
            arm_delta = np.concatenate([delta, delta_rot])
            action_dict = {"right": arm_delta, "right_gripper": np.array([gripper_state])}
            action = robot.create_action_vector(action_dict)
            print("action", action)
            obs, reward, done, info = env.step(action)

            # NEW: print gripper z position
            gripper_pos = robot._hand_pos

            eef_pos = gripper_pos["right"]
            eef_quat = robot._hand_quat["right"] # [x, y, z, w]

            joint_state = robot.sim.data.qpos[robot._ref_joint_pos_indexes] # [7]
            gripper_indices = list(robot._ref_gripper_joint_pos_indexes.values())
            gripper_qpos = robot.sim.data.qpos[gripper_indices]

            gripper_pos_m = np.array([np.mean(gripper_qpos)])

            small_obs = {k: v for k,v in obs.items()
                            if not k.endswith("_image") and not k.endswith("_depth") and not k.endswith("_segmentation_class")}

            log_entry = {
                "step":        step,
                "observation": convert_obs(small_obs),
                "reward": float(reward),
                "done": bool(done),
                "info": convert_obs(info),
                "action": action.tolist(),
            }

            known_objects = [obj.lower() for obj in INSTRUCTION_TEMPLATES.keys()]
            for key, value in log_entry["observation"].items():
                if not key.endswith("_pos"):
                    continue

                object_name = key.replace("_pos", "").lower()
                if object_name not in known_objects:
                    continue

                if isinstance(value, list) and len(value) > 1:
                    y_pos = value[1]
                    if y_pos >= 0.05 and gripper_state <= -1.0:
                        log_entry["done"] = True
                        break

            for cam in camera_names:
                image_key = f"{cam}_image"
                if image_key in obs:
                    rgb_img = obs[image_key]
                    bgr_img = cv2.cvtColor(rgb_img, cv2.COLOR_RGB2BGR)
                    bgr_img = cv2.flip(bgr_img, 0)
                    image_filename = os.path.join(base_dir, cam, f"{step:05d}.png")
                    cv2.imwrite(image_filename, bgr_img)
                    # if cv2.imwrite(image_filename, bgr_img):
                    #     #print(f"Saved RGB image: {image_filename}")
                    # else:
                    #     print(f"Failed to save RGB image: {image_filename}")
                    log_entry[image_key] = image_filename

                depth_key = f"{cam}_depth"
                if depth_key in obs:
                    depth_img = obs[depth_key]
                    d_min, d_max = depth_img.min(), depth_img.max()
                    if d_max - d_min > 1e-6:
                        depth_normalized = (depth_img - d_min) / (d_max - d_min) * 255
                    else:
                        depth_normalized = depth_img * 0
                    depth_normalized = depth_normalized.astype(np.uint8)
                    depth_normalized = cv2.flip(depth_normalized, 0)
                    depth_filename = os.path.join(base_dir, f"{cam}_depth", f"{step:05d}.png")
                    cv2.imwrite(depth_filename, depth_normalized)
                    # if cv2.imwrite(depth_filename, depth_normalized):
                    #     #print(f"Saved depth image: {depth_filename}")
                    # else:
                    #     print(f"Failed to save depth image: {depth_filename}")
                    log_entry[depth_key] = depth_filename

                seg_key = f"{cam}_segmentation_class"
                if seg_key in obs:
                    seg_mask = obs[seg_key]
                    
                    # Remove extra dimension if shape is (H, W, 1)
                    if seg_mask.ndim == 3 and seg_mask.shape[-1] == 1:
                        seg_mask = seg_mask[:, :, 0]
                    
                    seg_mask = cv2.flip(seg_mask, 0).astype(np.uint8)  # Flip + cast to uint8 for saving
                    seg_scaled = (seg_mask * (255 // (seg_mask.max() + 1))).astype(np.uint8)

                    # Apply color map (e.g., JET or HSV for diverse colors)
                    seg_color = cv2.applyColorMap(seg_scaled, cv2.COLORMAP_HSV)
                    mask_dir = os.path.join(base_dir, f"{cam}_segmentation")
                    os.makedirs(mask_dir, exist_ok=True)
                    mask_filename = os.path.join(mask_dir, f"{step:05d}_segmentation_class.png")
                    cv2.imwrite(mask_filename, seg_color)
                    
                    # Log only the filename, not the array
                    log_entry[f"{seg_key}_file"] = mask_filename

                     
            log_entry["observation"]["state.pos_xyzquat_right"] = np.concatenate([eef_pos, eef_quat]).tolist()
            log_entry["observation"]["state.joint_state"] = joint_state.tolist()
            log_entry["observation"]["state.position_m"] = gripper_pos_m.tolist()

            log_entry["action.pos_xyzquat_right"] = np.concatenate([delta, R.from_rotvec(delta_rot).as_quat()]).tolist()
            log_entry["action.joint_state"] = arm_delta[:7].tolist()  # or full joint delta if available
            log_entry["action.position_m"] = [gripper_state]

            
            data_log.append(log_entry)

            step += 1



    except KeyboardInterrupt:
        print("Keyboard interrupt. Exiting.")
    
    finally:
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
            'data': data_log
        }
        with open(os.path.join(base_dir, 'teleop_demo.json'), 'w') as f:
            json.dump(output, f, indent=2)
        print(f"Saved {len(data_log)} steps to teleop_demo.json")

        instruction = get_instruction_from_path(base_dir)
        with open(os.path.join(base_dir, "instruction.json"), "w") as f:
            json.dump({"instruction": instruction}, f, indent=2)
        print(f"✅ Saved instruction: {instruction}")


        env.close()
        cv2.destroyAllWindows()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--task", type=str, default="blue_bin_multi_object_picking", help="Task to run")

    args = parser.parse_args()
    main(args.task)
