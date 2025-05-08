import os
import pandas as pd
import json
from glob import glob

parquet_dir = "/home/elisa/Documents/MasterThesis/mujoco_data/eef_teleop/data/chunk-000/"
parquet_files = glob(os.path.join(parquet_dir, "*.parquet"))
root_dir = "/home/elisa/Documents/MasterThesis/mujoco_data/teleop_dataset_eef"
json_files = glob(os.path.join(parquet_dir, "*.json"))

# Modify parquet files and save new versions
# for file in parquet_files:
#     df = pd.read_parquet(file)

#     # Ensure required columns exist
#     if 'observation.object_pos' in df and 'next.done' in df and 'action' in df:
#         # If 'action' is a column of dicts, expand it
#         if isinstance(df['action'].iloc[0], dict):
#             action_df = pd.json_normalize(df['action'])
#             for col in action_df.columns:
#                 df[f"action.{col}"] = action_df[col]

#         # Set 'done = True' based on condition
#         df['object_y'] = df['observation.object_pos'].apply(lambda x: x[1] if len(x) > 1 else None)
#         # Example: compare with action.position_m and find rows where y > 0.05 and gripper is closed
#         mask = (df['object_y'] > 0.05) & (df['action.position_m'] <= 0.4)

#         df.loc[mask, 'next.done'] = True

#         # Fix gripper action values
#         df['action.position_m'] = df['action.position_m'].replace({-0.4: -1.0, 0.4: 1.0})
#         if 'action.-1' in df.columns:
#             df['action.-1'] = df['action.-1'].replace({-0.4: -1.0, 0.4: 1.0})

#         # Save modified file
#         new_parquet_path = os.path.join(parquet_dir, "modified_demo_teleop.parquet")
#         df.to_parquet(file, index=False)
#         print(f"Saved modified parquet file to: {new_parquet_path}")

for subfolder in os.listdir(root_dir):
    subfolder_path = os.path.join(root_dir, subfolder)
    if os.path.isdir(subfolder_path):
        json_path = os.path.join(subfolder_path, "teleop_demo.json")
        if os.path.exists(json_path):
            print(f"Found: {json_path}")
            # Read the JSON data
            with open(json_path, 'r') as f:
                try:
                    data = json.load(f)
                except json.JSONDecodeError:
                    print(f"Skipping invalid JSON file: {json_path}")
                    continue

            changed = False
            for entry in data:
                action = entry.get("action", [])

                # Flatten if nested (e.g., [[...]] → [...])
                if isinstance(action, list) and len(action) == 1 and isinstance(action[0], list):
                    action = action[0]
                    entry["action"] = action  # overwrite with flat list
                    changed = True
                # action_pos  = entry.get("action.position_m", {})[0]
                # action = entry.get("action", {})
                
                # # if action_pos == -0.4:
                # #     action_pos = -1.0
                # #     entry["action.position_m"] = [action_pos]
                # #     changed = True
                # # elif action_pos == 0.4:
                # #     action_pos = 1.0
                # #     entry["action.position_m"] = [action_pos]            
                # #     changed = True

                
                # # if action[-1] == -0.4:
                # #     action[-1] = -1.0
                # #     entry["action"] = [action]
                # #     changed = True
                # # elif action[-1] == 0.4:
                # #     action[-1] = 1.0
                # #     entry["action"] = [action]
                # #     changed = True
                    
                
                # obj_pos = entry["observation"].get("Bread_pos", {})[1]

                # # Set done = True if box_pos > 0.05 and gripper closed
                # if obj_pos > 0.05 and (
                #     (action_pos == -1.0) or
                #     (action[-1] == -1.0)
                # ):
                #     entry["done"] = True
                #     changed = True
                # else: 
                #     entry["done"] = False
                #     changed = True

            # Save modified version
            if changed:
                output_path = os.path.join(subfolder_path, "modified_teleop_demo.json")
                with open(json_path, 'w') as f:
                    json.dump(data, f, indent=2)
                print(f"Modified and saved: {output_path}")