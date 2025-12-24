# Adapted from openai baselines: https://github.com/openai/baselines/blob/master/baselines/common/atari_wrappers.py
from datetime import datetime
from typing import Optional
from omegaconf import OmegaConf
from collections import namedtuple
import cv2
# import gymnasium as gym
import gymnasium
import gym
import numpy as np
from ding.envs import ScaledFloatFrameWrapper
from ding.utils.compression_helper import jpeg_data_compressor
from easydict import EasyDict
# from gymnasium.wrappers import RecordVideo
from gym.wrappers import RecordVideo
from zoo.shapes2d.env.shapes2d import shapes2d
import torch
from zoo.ocr.slate.slate import SLATE

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
    env = gym.make(config.env_id)
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

    if config.warp_frame:
        env = WarpFrame(env, width=config.observation_shape[1], height=config.observation_shape[2], grayscale=config.gray_scale)

    if config.scale:
        env = ScaledFloatFrameWrapper(env)

    config_ocr = OmegaConf.load(config.ocr_config_path)
    config_env = namedtuple('EnvConfig', ['obs_size', 'obs_channels'])(config.observation_shape[2], 3)
    slate = SLATE(config_ocr, config_env, observation_space=None, preserve_slot_order=True)
    state_dict = torch.load(config.checkpoint_path)["ocr_module_state_dict"]
    slate._module.load_state_dict(state_dict)
    slate.requires_grad_(False)
    slate.eval()

    slot_extractor = SlotExtractor(model=slate, device='cuda', name_model = 'SLATE')

    env = SlotExtractorWrapper(env, slot_extractor, config.num_slots, config.slot_dim)

    if config.game_wrapper:
        env = GameWrapper(env)

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


class GameWrapper(gym.Wrapper):
    """
    Overview:
        A wrapper to adapt the environment to the game interface.
    """

    def __init__(self, env: gym.Env):
        """
        Arguments:
            - env (:obj:`gym.Env`): The environment to wrap.
        """
        super().__init__(env)

    def legal_actions(self):
        return [_ for _ in range(self.env.action_space.n)]

class GymnasiumToGymWrapper(gym.Wrapper):
    """
    Overview:
        A wrapper class that adapts a Gymnasium environment to the Gym interface.
    Interface:
        ``__init__``, ``reset``, ``seed``
    Properties:
        - _seed (:obj:`int` or None): The seed value for the environment.
    """

    def __init__(self, env):
        """
        Overview:
            Initializes the GymnasiumToGymWrapper.
        Arguments:
            - env (:obj:`gymnasium.Env`): The Gymnasium environment to be wrapped.
        """

        assert isinstance(env, gymnasium.Env), type(env)
        super().__init__(env)
        self._seed = None

    def seed(self, seed):
        """
        Overview:
            Sets the seed value for the environment.
        Arguments:
            - seed (:obj:`int`): The seed value to use for random number generation.
        """
        self._seed = seed

    def reset(self, **kwargs):
        """
        Overview:
            Resets the environment and returns the initial observation.
        Returns:
            - observation (:obj:`Any`): The initial observation of the environment.
        """
        result = self.env.reset(**kwargs)
        if isinstance(result, tuple):
            obs, info = result
        else:
            obs = result
        return obs

    def step(self, action):
        obs, rew, terminated, truncated, info = self.env.step(action)
        done = terminated or truncated
        return obs, rew, done, info


def obs_to_tensor(obs, device):
    if len(obs.shape) == 4:
        return torch.Tensor(obs.transpose(0, 3, 1, 2)).to(device) / 255.0
    else:
        return torch.Tensor(obs).to(device)


class SlotExtractor:
    def __init__(self, model, device, name_model):
        self._model = model
        self._device = device
        self._model.to(device)
        self.name_model = name_model

    def __call__(self, images, prev_slots, to_numpy=True):
        if len(images.shape) == 3:
            batch_images = images[np.newaxis, ...]
        else:
            batch_images = images

        if prev_slots is not None and len(prev_slots.shape) == 2:
            batch_prev_slots = prev_slots[np.newaxis, ...]
        else:
            batch_prev_slots = prev_slots

        batch_images = obs_to_tensor(batch_images, self._device)
        if batch_prev_slots is not None:
            batch_prev_slots = obs_to_tensor(batch_prev_slots, self._device)

        if self.name_model == 'SLATE':
            slots = self._model._module._get_slots(batch_images, prev_slots=batch_prev_slots).detach()
        else:
            slots = self._model(batch_images, prev_slots=batch_prev_slots).detach()

        if len(images.shape) == 3:
            slots = slots[0]

        if to_numpy:
            slots = slots.cpu().numpy()

        return slots

class SlotExtractorWrapper(gym.Wrapper):
    """
    Wrapper uses SlotExtractor in order to extract slots from the input image.
    """

    def __init__(self, env, slot_extractor, num_slots, slot_dim):
        super().__init__(env)

        self.slot_extractor = slot_extractor
        self.observation_space = gym.spaces.Box(
            low=-np.inf, high=np.inf, shape=(num_slots, slot_dim), dtype=np.float32
        )
        self.prev_slots = None

    def _get_slots(self, frame, prev_slots=None):
        if prev_slots is None:
            prev_slots = self.slot_extractor(frame, prev_slots=None)

        return self.slot_extractor(frame, prev_slots=prev_slots)

    def reset(self):
        frame = self.env.reset()
        self.prev_slots = self._get_slots(frame, prev_slots=None)
        return self.prev_slots.copy()

    def step(self, action):
        frame, reward, done, info = self.env.step(action)
        self.prev_slots = self._get_slots(frame, prev_slots=self.prev_slots)
        return self.prev_slots.copy(), reward, done, info