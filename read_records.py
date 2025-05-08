import json 

with open("/home/elisa/Documents/MasterThesis/mujoco_data/robosuite/panda_pickplacebread_demo.json", "r") as f:
    for i in range(0, 100):
        print(f.readline(), end="")
        print(f.keys())