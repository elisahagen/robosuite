import os
import xml.etree.ElementTree as ET
import trimesh

# 1) Point this at your MJCF asset XML
xml_path = "/home/elisa/Documents/masterthesis/robosuite/robosuite/models/assets/objects/bread_asset.xml"

# 2) Parse the <mesh> element
tree = ET.parse(xml_path)
root = tree.getroot()
mesh_elem = root.find(".//mesh")           # finds: <mesh file="meshes/bread.stl" name="bread_mesh" scale="0.8 0.8 0.8"/>

mesh_relpath = mesh_elem.attrib["file"]    # "meshes/bread.stl"
scale_factors = [float(s) for s in mesh_elem.attrib.get("scale", "1 1 1").split()]

# 3) Compute the absolute path to the STL
xml_dir = os.path.dirname(xml_path)
mesh_path = os.path.join(xml_dir, mesh_relpath)

# 4) Load, apply scale, and export as PLY
mesh = trimesh.load(mesh_path, force="mesh")          # loads the .stl
mesh.apply_scale(scale_factors)                       # applies the 0.8×0.8×0.8 scale
out_ply = os.path.join(xml_dir, "bread_canonical.ply")
mesh.export(out_ply, file_type="ply", encoding="ascii")

print(f"Wrote canonical mesh → {out_ply}")
