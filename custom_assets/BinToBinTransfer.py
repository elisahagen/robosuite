import numpy as np

from robosuite.environments.manipulation.pick_place import PickPlace
from robosuite.utils.placement_samplers import SequentialCompositeSampler, UniformRandomSampler
from robosuite.models.objects import MilkObject, BreadObject, BottleObject, CerealObject
from robosuite.models.objects import BoxObject, CapsuleObject, BallObject, CylinderObject


# # Define two cube objects
cube1 = BoxObject(name="cube1", size=[0.04, 0.04, 0.04], rgba=[1, 0, 0, 1]) 
cube2 = BoxObject(name="cube2", size=[0.04, 0.04, 0.04], rgba=[0, 1, 0, 1]) 
cube1_pos = [0.1 - 0.05, 0.28, 0.82]  
cube2_pos = [0.1 + 0.05, 0.28, 0.82]

class BinToBinTransfer(PickPlace):
    def __init__(self, target_obj, **kwargs):
        super().__init__(single_object_mode=0,**kwargs)

        self.target_obj = target_obj

    def _construct_objects(self):
        from robosuite.models.objects import BreadObject
        from robosuite.models.objects import BoxObject
        # self.objects = [BoxObject(name="Box", size=[0.02, 0.02, 0.02])] #, BreadObject(name="Bread2"), MilkObject(name="Milk"), BottleObject(name="Bottle")]
        self.objects = [BreadObject(name="Bread")] #, BreadObject(name="Bread2"), MilkObject(name="Milk"), BottleObject(name="Bottle")]
        # self.objects2 = [cube1, cube2]
    
    def _reset_internal(self):
        super()._reset_internal()  # resets objects, robot, etc.

        # Custom: randomize bin2 cubes here
        self.randomize_bin2_cubes()

    def randomize_bin2_cubes(self):
        # Bounds of bin2 in world space
        fixed_positions = [
            [ 0.15, 0.2, 0.9],  # position 1
            [ 0.05, 0.2, 0.9],  # position 2
            [ 0.15, 0.1, 0.9],  # position 3
            [ 0.05, 0.1, 0.9],  # position 4
        ]

        cube_names = ["cube1", "cube2", "cube3"]
        cube_colors = {
            "cube1": [1, 0, 0, 1],  # red
            "cube2": [0, 1, 0, 1],  # green
            "cube3": [0, 0, 1, 1],  # blue
            "cube4": [1, 1, 0, 1],  # yellow (or repeat any)
        }

        # Randomly assign them to positions
        assigned_positions = np.random.permutation(fixed_positions)[:3]

        self.cube_positions = {}
        self.used_positions = []

        for cube_name, pos in zip(cube_names, assigned_positions):
            body_id = self.sim.model.body_name2id(cube_name)
            self.sim.model.body_pos[body_id] = pos

            # Also set color (optional)
            geom_id = self.sim.model.geom_name2id(f"{cube_name}_geom")
            self.sim.model.geom_rgba[geom_id] = cube_colors[cube_name]

            self.cube_positions[cube_name] = pos
            self.used_positions.append(pos)

        # Optionally hide the 4th cube (move below ground)
        unplaced = list(set(cube_names) - set(cube_names))
        for cube_name in unplaced:
            body_id = self.sim.model.body_name2id(cube_name)
            self.sim.model.body_pos[body_id] = [0, 0, -1]  # hide below table
            self.cube_positions[cube_name] = [0, 0, -1]


        self.sim.forward()
        self.target_position = None
        for pos in fixed_positions:
            if not any(np.allclose(pos, used, atol=1e-4) for used in self.used_positions):
                self.target_position = pos
                break

    def _construct_visual_objects(self):
        self.visual_objects = []  # no visual goal objects

    def _get_placement_initializer(self):
        self.placement_initializer = SequentialCompositeSampler(name="ObjectSampler")
        bin_x_half = 0.08 #self.model.mujoco_arena.table_full_size[0] / 2 - 0.05
        bin_y_half = 0.08 #self.model.mujoco_arena.table_full_size[1] / 2 - 0.05

        self.placement_initializer.append_sampler(
            UniformRandomSampler(
                name="Bin1ObjectSampler",
                mujoco_objects=self.objects,
                x_range=[-0.025, 0.025],
                y_range=[-0.035, 0.145],
                rotation=self.z_rotation,
                rotation_axis="z",
                ensure_object_boundary_in_range=True,
                ensure_valid_placement=True,
                reference_pos=self.bin1_pos,
                z_offset=0.101,
            )
        )

        # self.placement_initializer.append_sampler(
        #     UniformRandomSampler(
        #         name="Bin2Cube1Sampler",
        #         mujoco_objects=self.objects2,
        #         x_range=[-bin_x_half, bin_x_half],
        #         y_range=[-bin_y_half, bin_y_half],
        #         rotation=self.z_rotation,
        #         rotation_axis="z",
        #         ensure_object_boundary_in_range=True,
        #         ensure_valid_placement=True,
        #         reference_pos=self.bin1_pos,
        #         z_offset=self.z_offset,
        #     )
        # )

    def _check_success(self):
        obj = self.objects[0]
        obj_pos = self.sim.data.body_xpos[self.obj_body_id[obj.name]]
        return not self.not_in_bin(obj_pos, bin_id=1)  # bin2
