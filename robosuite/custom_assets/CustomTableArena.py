import robosuite as suite
from robosuite.models.arenas import TableArena, MultiTableArena
from robosuite.models.objects import BoxObject
from robosuite.models.tasks import ManipulationTask
from robosuite.environments.manipulation.lift import Lift
from robosuite.models.objects import BallObject
import numpy as np
import copy
from robosuite.utils.mjcf_utils import array_to_string
import xml.etree.ElementTree as ET  # if needed for XML iteration

class CustomTableArena(TableArena):
    def __init__(self, custom_objects=None, **kwargs):

        super().__init__(**kwargs)
        cube1 = BoxObject(name="cube1", size=[0.05, 0.05, 0.05], rgba=[1.0, 0.0, 0.0, 1.0])
        cube2 = BoxObject(name="cube2", size=[0.05, 0.05, 0.05], rgba=[0.0, 1.0, 0.0, 1.0])
        custom_objects = custom_objects if custom_objects is not None else [cube1, cube2]
        self.custom_objects = custom_objects

        delta = (self.table_full_size[0] + 0.15) / 2.0

        # Set the first table's position so that it is shifted to the left by delta.
        first_table_center = np.array(self.center_pos) - np.array([0, delta, 0])
        self.table_body.set("pos", array_to_string(first_table_center))

        # --- Create and configure the second table ---
        second_table = copy.deepcopy(self.table_body)
        second_table.set("name", "table2")
        for geom in second_table.findall(".//geom"):
            orig_name = geom.get("name")
            if orig_name is not None:
                geom.set("name", orig_name + "_table2")
        for site in second_table.findall(".//site"):
            orig_name = site.get("name")
            if orig_name is not None:
                site.set("name", orig_name + "_table2")
        second_table_center = np.array(self.center_pos) + np.array([0, delta-0.1, 0])
        second_table.set("pos", array_to_string(second_table_center))
        self.second_table = second_table
        self.worldbody.append(second_table)

        sphere = BoxObject(
            name="cube5",
            size=[0.05, 0.05, 0.05],
            rgba=[0, 0.5, 0.5, 1]
        ).get_obj()
        sphere.set("pos", array_to_string(second_table_center))
        self.worldbody.append(sphere)

class CustomMultiTableArena(MultiTableArena):
    def __init__(self, **kwargs):
        # Create two table offsets: left and right
        table_offsets = [
            [-0.4, 0.0, 0.8],  # Left table
            [0.4, 0.0, 0.8],   # Right table
        ]
        super().__init__(table_offsets=table_offsets, **kwargs)

        # Add a fallback for environments expecting a single-table offset
        self.table_offset = np.array(table_offsets[0])

        # Add objects manually
        cube1 = BoxObject(name="cube1", size=[0.04, 0.04, 0.04], rgba=[1.0, 0.0, 0.0, 1.0])
        cube2 = BoxObject(name="cube2", size=[0.04, 0.04, 0.04], rgba=[0.0, 1.0, 0.0, 1.0])

        # Place cube2 on the second table
        cube2_obj = cube2.get_obj()
        cube2_obj.set("pos", array_to_string([0.4, 0.0, 0.85]))  # slight lift above table
        self.worldbody.append(cube2_obj)

        # Keep cube1 in self.custom_objects so it can be passed to the environment
        self.custom_objects = [cube1]

class CustomLift(Lift):
    def _load_model(self):

        super()._load_model()  
        self.custom_arena = CustomTableArena(
            table_full_size=self.table_full_size,
            table_friction=self.table_friction,
            table_offset=self.table_offset,
        )
        self.custom_arena = CustomMultiTableArena(
            table_full_sizes=(0.6, 0.6, 0.05),
            table_frictions=(1.0, 0.005, 0.0001),
        )
        self.model = ManipulationTask(
            mujoco_arena=self.custom_arena,
            mujoco_robots=[robot.robot_model for robot in self.robots],
            mujoco_objects=self.cube,
        )

        # first_table_pos = self.custom_arena.table_offsets[0]
        # cube_start_pos = np.array(first_table_pos) + np.array([0.0, 0.0, 0.07])  # slightly above table

        # self.cube.get_obj().set("pos", array_to_string(cube_start_pos))
