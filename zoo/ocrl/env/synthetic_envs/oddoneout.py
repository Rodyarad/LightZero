import numpy as np
from gym import spaces
from matplotlib import colors
from spriteworld import renderers as spriteworld_renderers
from spriteworld.sprite import Sprite

from .base import BaseEnv
from .utils import norm


class OddOneOutEnv(BaseEnv):
    def __init__(self, config, seed):
        super(OddOneOutEnv, self).__init__(config, seed)
        self._target_obj_idx = None
        self._unseen_combi_mode = config.unseen_combi_mode
        self._unseen_combi = config.unseen_combi
        self._obj_comp = config.obj_comp

    def _fill_properties(self, objs, unique_property, properties, idx):
        while sum(objs[:, idx] == 0) > 0:
            prop = self.np_random.choice(properties)
            while prop == unique_property:
                prop = self.np_random.choice(properties)
            if self._unseen_combi_mode is not None:
                if idx == 0:
                    found = False
                    while not found:
                        if self._unseen_combi_mode == "train":
                            if prop == unique_property:
                                prop = self.np_random.choice(properties)
                                found = False
                            elif unique_property == self._unseen_combi[0] and prop == self._unseen_combi[1]:
                                prop = self.np_random.choice(properties)
                                found = False
                            elif unique_property == self._unseen_combi[1] and prop == self._unseen_combi[0]:
                                prop = self.np_random.choice(properties)
                                found = False
                            else:
                                found = True
                        elif self._unseen_combi_mode == "test":
                            if prop == unique_property:
                                prop = self.np_random.choice(properties)
                                found = False
                            elif unique_property == self._unseen_combi[0] and prop == self._unseen_combi[1]:
                                found = True
                            elif unique_property == self._unseen_combi[1] and prop == self._unseen_combi[0]:
                                found = True
                            else:
                                prop = self.np_random.choice(properties)
                                found = False
                        else:
                            raise ValueError
            #if idx == 0:  # to validate more objects odd-one-out-N?C3S1S1
            #    num_assigned_objs = sum(objs[:, idx] == 0)
            #else:
            num_assigned_objs = self.np_random.integers(2, sum(objs[:, idx] == 0) + 1)
            while num_assigned_objs > 0:
                obj_idx = self.np_random.integers(len(objs))
                if objs[obj_idx, idx] == 0:
                    objs[obj_idx, idx] = prop
                    num_assigned_objs -= 1
            if sum(objs[:, idx] == 0) == 1:
                objs[objs[:, idx] == 0, idx] = prop
        return objs

    def _set_objs(self):
        objs = super()._set_objs()
        if self._unseen_combi_mode is not None:
            # when testing unseen combination, the target index is always 0 
            #  to modify the property selections through the rule for the unseen combination test
            target_obj_idx = 0
        else:
            target_obj_idx = self.np_random.integers(self._num_objects)
        # Randomly select the unique property
        types = []
        if len(self._COLORS) > 1:  # we cannot select the unique from a set of one item
            types.append("color")
        if len(self._SHAPES) > 1:
            types.append("shape")
        if len(self._SCALES) > 1:
            types.append("scale")
        rand_type = self.np_random.choice(types)
        if rand_type == "color":
            unique_property = self.np_random.choice(self._COLORS)
            if self._unseen_combi_mode == "test":
                found = False
                while not found:
                    if not unique_property in self._unseen_combi:
                        unique_property = self.np_random.choice(self._COLORS)
                    else:
                        found = True
            objs[target_obj_idx, 0] = unique_property
            if self._obj_comp:
                shape = self.np_random.choice(self._SHAPES)
                objs[:-1,1] = shape
                scale = self.np_random.choice(self._SCALES)
                objs[:-1,2] = scale
        elif rand_type == "shape":
            unique_property = self.np_random.choice(self._SHAPES)
            objs[target_obj_idx, 1] = unique_property
            if self._obj_comp:
                color = self.np_random.choice(self._COLORS)
                objs[:-1,0] = color
                scale = self.np_random.choice(self._SCALES)
                objs[:-1,2] = scale
        elif rand_type == "scale":
            unique_property = self.np_random.choice(self._SCALES)
            objs[target_obj_idx, 2] = unique_property
            if self._obj_comp:
                color = self.np_random.choice(self._COLORS)
                objs[:-1,0] = color
                shape = self.np_random.choice(self._SHAPES)
                objs[:-1,1] = shape
        # Assign other properties
        objs = self._fill_properties(objs, unique_property, self._COLORS, 0)
        objs = self._fill_properties(objs, unique_property, self._SHAPES, 1)
        objs = self._fill_properties(objs, unique_property, self._SCALES, 2)
        objs = self._fill_positions(
            objs,
            agent_eps=self._config.distance_to_agent,
            objs_eps=self._config.distance_to_objs,
            wall_eps=self._config.distance_to_wall,
        )

        self._target_obj_idx = target_obj_idx
        return objs

    def step(self, act):
        reward, is_success, done = super().step(act)
        reward, is_success, done = self._cal_reward(reward, is_success, done)
        return (
            self.render(),
            reward,
            done,
            {"is_success": is_success},
        )
