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
    def __init__(self, **kwargs):
        super().__init__(single_object_mode=0,**kwargs)

    def _construct_objects(self):
        from robosuite.models.objects import BreadObject
        from robosuite.models.objects import BoxObject
        # self.objects = [BoxObject(name="Box", size=[0.018, 0.018, 0.018])] #, BreadObject(name="Bread2"), MilkObject(name="Milk"), BottleObject(name="Bottle")]
        self.objects = [BreadObject(name="Bread")] #, BreadObject(name="Bread2"), MilkObject(name="Milk"), BottleObject(name="Bottle")]
        # self.objects2 = [cube1, cube2]
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
                x_range=[-0.03, 0.1],
                y_range=[-0.06, 0.2],
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
