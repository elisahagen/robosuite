import numpy as np
import robosuite as suite
import json
import os
import matplotlib.pyplot as plt
import cv2
from robosuite.models import MujocoWorldBase
from robosuite.models.robots import Panda
from robosuite.models.grippers import gripper_factory
from robosuite.models.arenas import TableArena
import mujoco
from mujoco import viewer 
from robosuite import load_composite_controller_config
from custom_assets.CustomTableArena import CustomTableArena, CustomLift


def convert_obs(obs):
    """
    Convert observations (and other nested data) to JSON‐serializable format.
    This recursively converts any numpy arrays to lists.
    """
    if isinstance(obs, np.ndarray):
        return obs.tolist()
    elif isinstance(obs, dict):
        return {key: convert_obs(val) for key, val in obs.items()}
    elif isinstance(obs, list):
        return [convert_obs(elem) for elem in obs]
    else:
        return obs


def main():
    base_dir = os.path.join(os.getcwd(), "images")
    
    # Create the necessary subdirectories for each camera.
    for cam in ["agentview", "frontview"]:
        rgb_dir = os.path.join(base_dir, cam)
        depth_dir = os.path.join(base_dir, f"{cam}_depth")
        os.makedirs(rgb_dir, exist_ok=True)
        os.makedirs(depth_dir, exist_ok=True)
        print(f"Created directories: {rgb_dir} and {depth_dir}")

    controller_config = load_composite_controller_config(controller="BASIC")


    # Create the envaironment instance using robosuite's API.
    env = CustomLift(
        robots="Panda",
        controller_configs=controller_config,  # Use the configured IK controller
        has_renderer=True,               
        has_offscreen_renderer=True,     
        use_camera_obs=True,          
        render_camera='frontview',   
        render_collision_mesh=False, 
        render_visual_mesh=True,
        renderer='mujoco',
        camera_names=["agentview", "frontview"],  
        camera_heights=240,              
        camera_widths=320,               
        camera_depths=True,             
        control_freq=20,
    )
    print(env.robots)
    obs = env.reset()
    robot = env.robots[0]
    print(obs["robot0_joint_pos"])
    print("Initial observation keys:", obs.keys())
    
    if "robot0_eef_pos" not in obs or "cube_pos" not in obs:
        print("Observation does not contain required keys 'robot0_eef_pos' or 'cube_pos'.")
        env.close()
        return

    start_eef = np.array(obs["robot0_eef_pos"])
    target_obj = np.array(obs["cube_pos"])
    print("Start EEF position:", start_eef)
    print("Target object position:", target_obj)
    
    num_traj_steps = 80
    trajectory = np.linspace(start_eef, target_obj, num_traj_steps)
    print("Trajectory:", trajectory)


    # Data log list to store step-by-step data.
    data_log = []

    # Reset environment to get initial observation
    obs = env.reset()
    
    gripper_state = 0.0
    close_threshold = 0.02  # in meters (for the x, y differences)
    vertical_threshold = 0.001  # gripper must be at least this far above cube in z
    num_steps = 80
    kp = 10.0
    for t in range(num_steps):
        current_eef = np.array(obs["robot0_eef_pos"])
        cube_pos = np.array(obs["cube_pos"])
        
        # Compute the error (desired change) between cube and gripper.
        error = cube_pos - current_eef
        
        # Compute the control command (proportional control).
        command = kp * error
        # Build a 6D delta vector: first three from command, and zeros for orientation.
        arm_delta = np.concatenate([command, np.zeros(3)])
        
        # Check differences in every axis.
        diff = np.abs(error)
        # Check if x and y differences are below the close threshold
        # and if the gripper is vertically above the cube by at least vertical_threshold.
        if diff[0] < close_threshold and diff[1] < close_threshold and (current_eef[2] - cube_pos[2]) > vertical_threshold:
            gripper_state = min(gripper_state + 0.05, 0.5)  # Close gripper
            print(f"Step {t}: Gripper above cube. Error: {error}")
        else:
            if t <= 60:
                gripper_command = np.array([0.0])
                print(f"Step {t}: Moving towards cube. Error: {error}")
            else:
                print("not moving grupper")

        if t >= 60: 
            print("Lifting cube")
            arm_delta = np.concatenate([np.array([0.0, 0.0, 0.5]), np.zeros(3)])
        # Create and execute the action.
        action_dict = {
            "right": arm_delta,
            "right_gripper": gripper_command,
        }
    
        action = robot.create_action_vector(action_dict)
        obs, reward, done, info = env.step(action)
        
        log_entry = {
            "step": t,
            "observation": convert_obs(obs),
            "reward": float(reward),
            "done": bool(done),
            "info": convert_obs(info),
            "action": action.tolist(),
        }

        # if t == 2:
        #     for cam in ["agentview", "frontview"]:
        #         image_key = f"{cam}_image"
        #         if image_key in obs:
        #             rgb_img = obs[image_key]
        #             print(f"{image_key} shape:", rgb_img.shape, "min:", rgb_img.min(), "max:", rgb_img.max())

        if t % 2 == 0: 
            for cam in ["agentview", "frontview"]:
                # Save RGB image.
                image_key = f"{cam}_image"
                if image_key in obs:
                    rgb_img = obs[image_key]
                    # Convert RGB (used in robosuite) to BGR for correct OpenCV saving.
                    bgr_img = cv2.cvtColor(rgb_img, cv2.COLOR_RGB2BGR)
                    bgr_img = cv2.flip(bgr_img, 0)
                    image_filename = os.path.join(base_dir, cam, f"{t:05d}.png")
                    if cv2.imwrite(image_filename, bgr_img):
                        print(f"Saved RGB image: {image_filename}")
                    else:
                        print(f"Failed to save RGB image: {image_filename}")
                    log_entry[image_key] = image_filename

                # Save depth image.
                depth_key = f"{cam}_depth"
                if depth_key in obs:
                    depth_img = obs[depth_key]
                    # Normalize the depth image for visualization.
                    d_min, d_max = depth_img.min(), depth_img.max()
                    if d_max - d_min > 1e-6:
                        depth_normalized = (depth_img - d_min) / (d_max - d_min) * 255
                    else:
                        depth_normalized = depth_img * 0
                    depth_normalized = depth_normalized.astype(np.uint8)
                    depth_normalized = cv2.flip(depth_normalized, 0)
                    depth_filename = os.path.join(base_dir, f"{cam}_depth", f"{t:05d}.png")
                    if cv2.imwrite(depth_filename, depth_normalized):
                        print(f"Saved depth image: {depth_filename}")
                    else:
                        print(f"Failed to save depth image: {depth_filename}")
                    log_entry[depth_key] = depth_filename
        data_log.append(log_entry)
        # Render the environment (optional)
        # env.render()

        # If the episode ends, reset the environment and log that event.
        if done:
            print(f"Episode finished after {t+1} steps, resetting environment.")
            obs = env.reset()
    
    # Save the recorded data to a JSON file
    output_file = "panda_pickplacebread_demo.json"
    with open(output_file, "w") as f:
        json.dump(data_log, f, indent=2)
    
    print(f"Recorded demonstration data saved to: {os.path.abspath(output_file)}")
    
    # Close the environment to properly shut down the viewer
    env.close()

if __name__ == "__main__":
    main()

