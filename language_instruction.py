# import os
# import json
# import random

# # Customizable phrases
# INSTRUCTION_TEMPLATES = {
#     "milk": [
#         "Please grab the milk and place it on the table.",
#         "Pick up the milk carton and move it to the tray.",
#         "Lift the milk and drop it in the basket.",
#         "Take the milk and place it near the sink."
#     ],
#     "bread": [
#         "Pick up the bread and put it in the basket.",
#         "Grab the loaf of bread and move it to the counter.",
#         "Lift the bread and drop it on the tray.",
#         "Please relocate the bread to the plate."
#     ],
#     "can": [
#         "Take the can and place it on the shelf.",
#         "Pick up the can and move it to the bin.",
#         "Grab the can and drop it near the box.",
#         "Relocate the can to the storage area."
#     ],
#     "juice": [
#         "Lift the juice bottle and set it on the table.",
#         "Pick up the juice and move it next to the milk.",
#         "Grab the juice container and drop it in the tray.",
#         "Take the juice and place it on the counter."
#     ],
#     "cereals": [
#         "Lift the cereal box and set it on the table.",
#         "Pick up the cereal box and move it next to the milk.",
#         "Grab the cereal box and drop it in the tray.",
#         "Take the cereal box and place it on the counter."
#     ],
#     "bottle": [
#         "Lift the bottle and set it on the table.",
#         "Pick up the bottle and move it next to the milk.",
#         "Grab the bottle and drop it in the tray.",
#         "Take the bottle and place it on the counter."
#     ]
# }

# def find_object_in_name(name: str):
#     """Extract object keyword from folder name"""
#     for obj in INSTRUCTION_TEMPLATES:
#         if obj in name.lower():
#             return obj
#     return None

# def create_instruction_json(root_folder):
#     for folder_name in os.listdir(root_folder):
#         print(folder_name)
#         folder_path = os.path.join(root_folder, folder_name)
#         if os.path.isdir(folder_path):
#             obj = find_object_in_name(folder_name)
#             if obj:
#                 instruction = random.choice(INSTRUCTION_TEMPLATES[obj])
#                 output_path = os.path.join(folder_path, "instruction.json")
#                 with open(output_path, "w") as f:
#                     json.dump({"instruction": instruction}, f, indent=2)
#                 print(f"✅ {folder_name}: '{instruction}'")
#             else:
#                 print(f"⚠ Skipping '{folder_name}': no known object found.")

# if __name__ == "__main__":
#     create_instruction_json("/home/elisa/Documents/MasterThesis/mujoco_data/teleop_panda_mujoco/")
# # Example usage:
# # create_instruction_json("/path/to/your/data/folder")


import os
import json
import random

root_dir = "/home/elisa/Documents/MasterThesis/mujoco_data/teleop_dataset_eef/"  
instruction_options = [
    "Lift the cube and set it on the table.",
    "Grab the cube and drop it in the tray.",
    "Take the cube and place it on the counter.",
    "Pick up the cube and put it on the shelf.",
    "Move the cube from the floor to the platform.",
    "Place the cube gently onto the marked area."
]

for subfolder in os.listdir(root_dir):
    if "cube" in subfolder.lower():
        subfolder_path = os.path.join(root_dir, subfolder)
        if os.path.isdir(subfolder_path):
            instruction_path = os.path.join(subfolder_path, "instruction.json")
            if os.path.exists(instruction_path):
                try:
                    with open(instruction_path, "r") as f:
                        instruction_data = json.load(f)

                    instruction_data["instruction"] = random.choice(instruction_options)

                    with open(instruction_path, "w") as f:
                        json.dump(instruction_data, f, indent=2)
                    
                    print(f"Updated instruction in: {instruction_path}")

                except json.JSONDecodeError:
                    print(f"Invalid JSON in: {instruction_path}")
