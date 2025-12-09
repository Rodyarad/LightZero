import gym
import numpy as np
import robosuite
from robosuite import load_controller_config
from robosuite.utils.placement_samplers import UniformRandomSampler, ObjectPositionSampler

from copy import copy


class FixedPositionSampler(ObjectPositionSampler):
    def __init__(self, name, task, mujoco_objects=None, ensure_object_boundary_in_range=True, ensure_valid_placement=True,
                 reference_pos=(0, 0, 0), z_offset=0.0):
        # Setup attributes
        super().__init__(name, mujoco_objects, ensure_object_boundary_in_range, ensure_valid_placement, reference_pos,
                         z_offset)

        assert task in ('Stack', 'Lift')

        if task == 'Stack':
            self.placement = {'cubeA': ((0.05, -0.15, 0.8300000000000001),
                                        np.array([-0.84408914, 0., 0., 0.53620288], dtype=np.float32)),
                              'cubeB': ((-0.05, 0.2, 0.8350000000000001),
                                        np.array([-0.85059733, 0., 0., 0.52581763], dtype=np.float32)),}
        else:
            self.placement = {
                'cube': ((0.12, 0.12, 0.8350000000000001), np.array([-0.5, 0., 0., 0.8], dtype=np.float32))}

    def sample(self, fixtures=None, reference=None, on_top=True):
        placed_objects = {} if fixtures is None else copy(fixtures)
        placement = copy(self.placement)
        # Sample pos and quat for all objects assigned to this sampler
        for obj in self.mujoco_objects:
            # First make sure the currently sampled object hasn't already been sampled
            assert obj.name not in placed_objects, obj.name
            assert obj.name in placement, obj.name

            placement[obj.name] = placement[obj.name] + (obj,)

        return placement


class RobosuiteEnv(gym.Env):
    metadata = {"render.modes": ["rgb_array"]}

    def __init__(self, task, horizon, seed, initialization_noise_magnitude=None, use_random_object_position=False):
        assert task in ('Lift', 'Stack')
        self._task = task
        self._horizon = horizon
        self._initialization_noise_magnitude = initialization_noise_magnitude
        self._use_random_object_position = use_random_object_position
        self._seed = seed
        self.render_mode = self.metadata["render.modes"][0]

        np.random.seed(self._seed)
        controller_config = load_controller_config(default_controller="OSC_POSITION")

        placement_initializer = FixedPositionSampler("ObjectSampler", self._task)
        if self._use_random_object_position == 'large':
            placement_initializer = UniformRandomSampler(
                name="ObjectSampler",
                x_range=[-0.25, 0.25],
                y_range=[-0.25, 0.25],
                rotation=None,
                ensure_object_boundary_in_range=False,
                ensure_valid_placement=True,
                reference_pos=np.array((0, 0, 0.8)),
                z_offset=0.01,
            )
        elif self._use_random_object_position == 'small':
            placement_initializer = UniformRandomSampler(
                name="ObjectSampler",
                x_range=[0.08, 0.12],
                y_range=[0.08, 0.12],
                rotation=None,
                ensure_object_boundary_in_range=False,
                ensure_valid_placement=True,
                reference_pos=np.array((0, 0, 0.8)),
                z_offset=0.01,
            )

        initialization_noise = None
        if self._initialization_noise_magnitude is not None:
            initialization_noise = {'magnitude': initialization_noise_magnitude, 'type': 'uniform'}

        camera_name = 'frontview'
        self._image_key_name = f'{camera_name}_image'
        env = robosuite.make(
            task,
            robots=["Panda"],
            gripper_types="default",
            controller_configs=controller_config,
            env_configuration="default",
            use_camera_obs=True,
            use_object_obs=False,
            reward_shaping=True,
            has_renderer=False,
            has_offscreen_renderer=True,
            control_freq=20,
            horizon=horizon,
            camera_names="frontview",
            placement_initializer=placement_initializer,
            initialization_noise=initialization_noise,
            camera_heights=256,
            camera_widths=256,
            ignore_done=False,
        )

        self._env = env
        self._last_frame = None
        self._crop = ((18, 202), (36, 220))
        observation_space = (self._crop[0][1] - self._crop[0][0], self._crop[1][1] - self._crop[1][0], 3)
        self.observation_space = gym.spaces.Box(0, 255, observation_space, dtype=np.uint8)

        low, high = self._env.action_spec
        self.action_space = gym.spaces.Box(low, high)

    def _process_observation(self, observation):
        return np.flipud(observation[self._image_key_name])[self._crop[0][0]:self._crop[0][1], self._crop[1][0]:self._crop[1][1]]

    def render(self, mode=None):
        return self._last_frame

    def reset(self, seed=None, options=None):
        self._last_frame = self._process_observation(self._env.reset())
        return self._last_frame

    def step(self, action):
        observation, reward, robosuite_done, info = self._env.step(action)
        self._last_frame = self._process_observation(observation)
        return self._last_frame, reward, robosuite_done, info
