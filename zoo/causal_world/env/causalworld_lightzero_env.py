import copy
import os
from datetime import datetime
from typing import Callable, Union, Dict, List
from typing import Optional
from easydict import EasyDict
import copy
import os
from datetime import datetime
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import gym
import numpy as np
from ding.envs import BaseEnv, BaseEnvTimestep
from ding.envs.common import save_frames_as_gif
from ding.envs.common.common_function import affine_transform
from ding.torch_utils import to_ndarray
from ding.utils import ENV_REGISTRY

from zoo.causal_world.env.causalworld_wrappers import wrap_lightzero


@ENV_REGISTRY.register('causalworld_lightzero')
class CausalWorldEnvLightZero(BaseEnv):

    @classmethod
    def default_config(cls: type) -> EasyDict:
        cfg = EasyDict(copy.deepcopy(cls.config))
        cfg.cfg_type = cls.__name__ + 'Dict'
        return cfg

    config = dict(
        seed=0,
        stop_value=int(1e6),
        from_pixels=True,
        observation_shape=(3, 84, 84),
        scale=True,
        warp_frame=True,
        channel_last=False,
        gray_scale=False,
        # replay_path (str or None): The path to save the replay video. If None, the replay will not be saved.
        # Only effective when env_manager.type is 'base'.
        replay_path=None,
        # (bool) If True, save the replay as a gif file.
        save_replay_gif=False,
        # (str or None) The path to save the replay gif. If None, the replay gif will not be saved.
        replay_path_gif=None,
        transform2string=False,
        env_config_path='zoo/causal_world/env/causal_world/cw_envs/config/reaching-hard_orig.yaml',
        collect_max_episode_steps=int(125),
        eval_max_episode_steps=int(125),
        norm_obs=dict(use_norm=False, ),
        norm_reward=dict(use_norm=False, ),
    )

    def __init__(self, cfg: dict) -> None:
        """
        Overview:
            Initialize the MuJoCo environment.
        Arguments:
            - cfg (:obj:`dict`): Configuration dict. The dict should include keys like 'env_id', 'replay_path', etc.
        """
        default_config = self.default_config()
        default_config.update(cfg)
        self._cfg = default_config
        self.channel_last = self._cfg.channel_last
        self._init_flag = False
        self._replay_path = None
        self._replay_path_gif = self._cfg.replay_path_gif
        self._save_replay_gif = self._cfg.save_replay_gif
        self._timestep = 0
        self._save_replay_count = 0

    def reset(self) -> Dict[str, np.ndarray]:
        """
        Reset the environment. If it hasn't been initialized yet, this method also handles that. It also handles seeding
        if necessary. Returns the first observation.
        """
        if not self._init_flag:
            self._env = wrap_lightzero(self._cfg)

            self._observation_space = gym.spaces.Dict({
                'observation': gym.spaces.Box(
                    low=0, high=1, shape=self._cfg.observation_shape, dtype=np.float32
                ),
                'action_mask': gym.spaces.Box(
                    low=0,
                    high=1,
                    shape=(1,),
                    dtype=np.int8
                ),
                'to_play': gym.spaces.Box(
                    low=-1,
                    high=2,
                    shape=(),
                    dtype=np.int8
                ),
                'timestep': gym.spaces.Box(
                    low=0,
                    high=self._cfg.collect_max_episode_steps,
                    shape=(),
                    dtype=np.int32
                ),
            })
            self._action_space = self._env.action_space
            self._reward_space = gym.spaces.Box(
                low=self._env.reward_range[0], high=self._env.reward_range[1], shape=(1,), dtype=np.float32
            )
            self._init_flag = True

        if hasattr(self, '_seed') and hasattr(self, '_dynamic_seed') and self._dynamic_seed:
            np_seed = 100 * np.random.randint(1, 1000)
            self._env.seed(self._seed + np_seed)
            self._env.action_space.seed(self._seed + np_seed)
        elif hasattr(self, '_seed'):
            self._env.seed(self._seed)
            self._env.action_space.seed(self._seed)
        

        obs = self._env.reset()
        obs = to_ndarray(obs).astype('float32')

        if self._cfg.from_pixels:
            if not self.channel_last:
                obs = np.transpose(obs, (2, 0, 1))

        self._eval_episode_return = 0.
        self._timestep = 0

        if self._save_replay_gif:
            self._frames = []

        action_mask = None
        obs = {'observation': obs, 'action_mask': np.array(action_mask), 'to_play': np.array(-1), 'timestep': np.array(self._timestep)}

        return obs

    def step(self, action: Union[np.ndarray, list]) -> BaseEnvTimestep:
        """
        Overview:
            Perform a step in the environment using the provided action, and return the next state of the environment.
            The next state is encapsulated in a BaseEnvTimestep object, which includes the new observation, reward,
            done flag, and info dictionary.
        Arguments:
            - action (:obj:`Union[np.ndarray, list]`): The action to be performed in the environment. 
        Returns:
            - timestep (:obj:`BaseEnvTimestep`): An object containing the new observation, reward, done flag,
              and info dictionary.
        .. note::
            - The cumulative reward (`_eval_episode_return`) is updated with the reward obtained in this step.
            - If the episode ends (done is True), the total reward for the episode is stored in the info dictionary
              under the key 'eval_episode_return'.
            - An action mask is created with ones, which represents the availability of each action in the action space.
            - Observations are returned in a dictionary format containing 'observation', 'action_mask', and 'to_play'.
        """
        action = action.astype('float32')
        action = affine_transform(action, min_val=self._env.action_space.low, max_val=self._env.action_space.high)

        if self._save_replay_gif:
            self._frames.append(self._env.render(mode='rgb_array'))
        obs, rew, done, info = self._env.step(action)
        self._timestep += 1
        self._eval_episode_return += rew

        if self._timestep > self._cfg.max_episode_steps:
            done = True

        if done:
            info['eval_episode_return'] = self._eval_episode_return
            if self._save_replay_gif:

                if not os.path.exists(self._replay_path_gif):
                    os.makedirs(self._replay_path_gif)
                timestamp = datetime.now().strftime("%Y%m%d%H%M%S")
                path = os.path.join(
                    self._replay_path_gif,
                    '{}_episode_{}_seed{}_{}.gif'.format(f'{self._cfg["domain_name"]}_{self._cfg["task_name"]}', self._save_replay_count, self._seed, timestamp)
                )
                self.display_frames_as_gif(self._frames, path)
                print(f'save episode {self._save_replay_count} in {self._replay_path_gif}!')
                self._save_replay_count += 1

        obs = to_ndarray(obs).astype(np.float32)
        rew = to_ndarray(rew).astype(np.float32)

        if self._cfg.from_pixels:
            if not self.channel_last:
                obs = np.transpose(obs, (2, 0, 1))

        action_mask = None
        obs = {'observation': obs, 'action_mask': np.array(action_mask), 'to_play': np.array(-1), 'timestep': np.array(self._timestep)}

        return BaseEnvTimestep(obs, rew, done, info)

    def close(self) -> None:
        """
        Close the environment, and set the initialization flag to False.
        """
        if self._init_flag:
            self._env.close()
        self._init_flag = False

    def seed(self, seed: int, dynamic_seed: bool = True) -> None:
        """
        Set the seed for the environment's random number generator. Can handle both static and dynamic seeding.
        """
        self._seed = seed
        self._dynamic_seed = dynamic_seed
        np.random.seed(self._seed)

    def enable_save_replay(self, replay_path: Optional[str] = None) -> None:
        """
        Enable saving replay videos to the specified path.
        """
        if replay_path is None:
            replay_path = './video'
        self._replay_path = replay_path

    @property
    def observation_space(self) -> gym.spaces.Space:
        """
        Property to access the observation space of the environment.
        """
        return self._observation_space

    @property
    def action_space(self) -> gym.spaces.Space:
        """
        Property to access the action space of the environment.
        """
        return self._action_space

    @property
    def reward_space(self) -> gym.spaces.Space:
        """
        Property to access the reward space of the environment.
        """
        return self._reward_space

    def __repr__(self) -> str:
        """
        String representation of the environment.
        """
        return "LightZero CausalWorld Env reaching-hard"

    @staticmethod
    def display_frames_as_gif(frames: list, path: str) -> None:
        frames = [np.transpose(frame, (1, 2, 0)) for frame in frames]

        patch = plt.imshow(frames[0])
        plt.axis('off')

        def animate(i):
            patch.set_data(frames[i])

        anim = animation.FuncAnimation(plt.gcf(), animate, frames=len(frames), interval=5)
        anim.save(path, writer='pillow', fps=20)

    def random_action(self) -> np.ndarray:
        """
        Generate a random action using the action space's sample method. Returns a numpy array containing the action.
        """
        random_action = self.action_space.sample().astype(np.float32)
        return random_action

    @staticmethod
    def create_collector_env_cfg(cfg: dict) -> List[dict]:
        collector_env_num = cfg.pop('collector_env_num')
        cfg = copy.deepcopy(cfg)
        cfg.max_episode_steps = cfg.collect_max_episode_steps
        cfg.is_eval = False
        return [cfg for _ in range(collector_env_num)]

    @staticmethod
    def create_evaluator_env_cfg(cfg: dict) -> List[dict]:
        evaluator_env_num = cfg.pop('evaluator_env_num')
        cfg = copy.deepcopy(cfg)
        cfg.max_episode_steps = cfg.eval_max_episode_steps
        cfg.is_eval = True
        return [cfg for _ in range(evaluator_env_num)]

