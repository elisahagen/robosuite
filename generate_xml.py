import xml.etree.ElementTree as ET
from robosuite.models.objects import BoxObject

# create a cube of side length 0.06 m
cube = BoxObject(name="cube", 
                 size_min=[0.02, 0.02, 0.02], 
                 size_max=[0.02, 0.02, 0.02])

# get the root <body> subtree for the cube:
cube_subtree = cube.get_obj()  

# wrap in a minimal <mujoco> tag if you like:
mujoco_root = ET.Element("mujoco", model="cube")
assets = ET.SubElement(mujoco_root, "asset")
# (optionally add material/texture tags here)
worldbody = ET.SubElement(mujoco_root, "worldbody")
worldbody.append(cube_subtree)

tree = ET.ElementTree(mujoco_root)
tree.write("cube.xml", encoding="utf-8", xml_declaration=True)

import trimesh
import os

base_dir = "/home/elisa/Documents/masterthesis/git/robosuite/robosuite/models/assets/objects"
# robosuite BoxObject stores half-extents in cube.size; double them for full edge length
half_extents = cube.size  # e.g. [0.03,0.03,0.03]
full_extents = [2*x for x in half_extents]

# create a trimesh box
mesh = trimesh.creation.box(extents=full_extents)

# export exactly like you did for bread
out_path = os.path.join(base_dir, "cube_canonical.ply")
mesh.export(out_path, file_type="ply", encoding="ascii")
print(f"[+] wrote canonical cube mesh → {out_path}")