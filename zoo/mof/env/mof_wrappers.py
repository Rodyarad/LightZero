# Adapted from openai baselines: https://github.com/openai/baselines/blob/master/baselines/common/atari_wrappers.py
from datetime import datetime
from typing import Optional

import cv2
# import gymnasium as gym
import gym
import numpy as np
from ding.envs import ScaledFloatFrameWrapper
from ding.utils.compression_helper import jpeg_data_compressor
from easydict import EasyDict
# from gymnasium.wrappers import RecordVideo
from gym.wrappers import RecordVideo

from omegaconf import OmegaConf
import multi_object_fetch
from multi_object_fetch.env import MultiObjectFetchEnv
from PIL import Image
from typing import Tuple, Callable
import random

def wrap_lightzero(config: EasyDict) -> gym.Env:
    """
    Overview:
        Configure environment for MuZero-style Shapes2d. The observation is
        channel-first: (c, h, w) instead of (h, w, c).
    Arguments:
        - config (:obj:`Dict`): Dict containing configuration parameters for the environment.
        - episode_life (:obj:`bool`): If True, the agent starts with a set number of lives and loses them during the game.
        - clip_rewards (:obj:`bool`): If True, the rewards are clipped to a certain range.
    Return:
        - env (:obj:`gym.Env`): The wrapped Atari environment with the given configurations.
    """
    task, distractors, reward = config.env_id.split('_')
    min_distractors, max_distractors = map(int, distractors[:-len('Distractors')].split('to'))
    env_names = []
    for num_distractors in range(min_distractors, max_distractors + 1):
        env_names.append(f'{task}_{num_distractors}Distractors_{reward}')

    def create_random_env():
        name = random.choice(env_names)
        return make_env(config,
                        name,
                        (config.observation_shape[1], config.observation_shape[2]), 
                        config.max_episode_steps, 
                        config.action_repeat, 
                        config.seed)
    env = VariableDistractorsWrapper(create_random_env(), create_random_env)

    return env


class TimeLimit(gym.Wrapper):
    """
    Overview:
        A wrapper that limits the maximum number of steps in an episode.
    """

    def __init__(self, env: gym.Env, max_episode_steps: Optional[int] = None):
        """
        Arguments:
            - env (:obj:`gym.Env`): The environment to wrap.
            - max_episode_steps (:obj:`Optional[int]`): Maximum number of steps per episode. If None, no limit is applied.
        """
        super(TimeLimit, self).__init__(env)
        self._max_episode_steps = max_episode_steps
        self._elapsed_steps = 0

    def step(self, ac):
        observation, reward, done, info = self.env.step(ac)
        self._elapsed_steps += 1
        if self._elapsed_steps >= self._max_episode_steps:
            done = True
            info['TimeLimit.truncated'] = True
        return observation, reward, done, info

    def reset(self, **kwargs):
        self._elapsed_steps = 0
        return self.env.reset(**kwargs)


class WarpFrame(gym.ObservationWrapper):
    """
    Overview:
        A wrapper that warps frames to 84x84 as done in the Nature paper and later work.
    """

    def __init__(self, env: gym.Env, width: int = 84, height: int = 84, grayscale: bool = True,
                 dict_space_key: Optional[str] = None):
        """
        Arguments:
            - env (:obj:`gym.Env`): The environment to wrap.
            - width (:obj:`int`): The width to which the frames are resized.
            - height (:obj:`int`): The height to which the frames are resized.
            - grayscale (:obj:`bool`): If True, convert frames to grayscale.
            - dict_space_key (:obj:`Optional[str]`): If specified, indicates which observation should be warped.
        """
        super().__init__(env)
        self._width = width
        self._height = height
        self._grayscale = grayscale
        self._key = dict_space_key
        if self._grayscale:
            num_colors = 1
        else:
            num_colors = 3

        new_space = gym.spaces.Box(
            low=0,
            high=255,
            shape=(self._height, self._width, num_colors),
            dtype=np.uint8,
        )
        if self._key is None:
            original_space = self.observation_space
            self.observation_space = new_space
        else:
            original_space = self.observation_space.spaces[self._key]
            self.observation_space.spaces[self._key] = new_space
        assert original_space.dtype == np.uint8 and len(original_space.shape) == 3

    def observation(self, obs):
        if self._key is None:
            frame = obs
        else:
            frame = obs[self._key]

        if self._grayscale:
            frame = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)
        frame = cv2.resize(frame, (self._width, self._height), interpolation=cv2.INTER_AREA)
        if self._grayscale:
            frame = np.expand_dims(frame, -1)

        if self._key is None:
            obs = frame
        else:
            obs = obs.copy()
            obs[self._key] = frame
        return obs

class JpegWrapper(gym.Wrapper):
    """
    Overview:
        A wrapper that converts the observation into a string to save memory.
    """

    def __init__(self, env: gym.Env, transform2string: bool = True):
        """
        Arguments:
            - env (:obj:`gym.Env`): The environment to wrap.
            - transform2string (:obj:`bool`): If True, transform the observations to string.
        """
        super().__init__(env)
        self.transform2string = transform2string

    def step(self, action):
        observation, reward, done, info = self.env.step(action)

        if self.transform2string:
            observation = jpeg_data_compressor(observation)

        return observation, reward, done, info

    def reset(self, **kwargs):
        observation = self.env.reset(**kwargs)

        if self.transform2string:
            observation = jpeg_data_compressor(observation)

        return observation


class Pixels(gym.Wrapper):
    def __init__(self, env: gym.Env, image_size: Tuple[int, int]) -> None:
        super().__init__(env)
        self.image_size = image_size
        self.observation_space = gym.spaces.Box(low=0, high=1, shape=(3,) + image_size, dtype=float)

    def step(self, action: np.ndarray) -> Tuple[np.ndarray, float, bool, dict]:
        _, reward, done, info = self.env.step(action)
        return self._get_obs(), reward, done, info

    def reset(self) -> np.ndarray:
        self.env.reset()
        return self._get_obs()

    def _get_obs(self) -> np.ndarray:
        if isinstance(self.env.unwrapped, MultiObjectFetchEnv):
            image = self.env.render(mode='rgb_array', size=self.image_size)
        else:
            image = Image.fromarray(self.env.render(mode='rgb_array'))
            image = np.array(image.resize(self.image_size))
        return np.moveaxis(image, -1, 0)

class ActionRepeat(gym.Wrapper):
    def __init__(self, env: gym.Env, action_repeat: int = 1) -> None:
        super().__init__(env)
        self.action_repeat = action_repeat

    def step(self, action: np.ndarray) -> Tuple[np.ndarray, float, bool, dict]:
        accumulated_reward = 0.0
        for _ in range(self.action_repeat):
            obs, reward, done, info = self.env.step(action)
            accumulated_reward += reward
            if done:
                break
        return obs, accumulated_reward, done, info

class VariableDistractorsWrapper(gym.Wrapper):
    def __init__(self, env: gym.Env, create_random_env: Callable) -> None:
        super().__init__(env)
        self.create_random_env = create_random_env

    def reset(self) -> np.ndarray:
        self.env = self.create_random_env()
        return self.env.reset()

def make_env(config: EasyDict, name: str, image_size: Tuple[int, int], max_episode_steps: int, action_repeat: int, seed: int = 0):
    env = gym.make(name, disable_env_checker=True)
    if hasattr(config, 'save_replay') and config.save_replay \
            and hasattr(config, 'replay_path') and config.replay_path is not None:
        timestamp = datetime.now().strftime("%Y%m%d%H%M%S")
        video_name = f'{env.spec.id}-video-{timestamp}'
        env = RecordVideo(
            env,
            video_folder=config.replay_path,
            episode_trigger=lambda episode_id: True,
            name_prefix=video_name
        )
    env = ActionRepeat(env, action_repeat)
    env = TimeLimit(env, max_episode_steps)
    env = Pixels(env, image_size)
    env = JpegWrapper(env, transform2string=config.transform2string)
    return env