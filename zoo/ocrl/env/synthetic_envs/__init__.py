import gym
from pathlib import Path

from omegaconf import OmegaConf

from .oddoneout import OddOneOutEnv
from .randomobjs import RandomObjsEnv
from .target import TargetEnv
from .push import PushEnv
from .maze import MazeEnv


_CONFIG_DIR = Path(__file__).resolve().parent.parent / "env_configs"


def _load_synthetic_env_cfg(config_stem: str):

    base_cfg = OmegaConf.load(_CONFIG_DIR / "_synthetic_env_base.yaml")
    env_cfg = OmegaConf.load(_CONFIG_DIR / f"{config_stem}.yaml")
    cfg = OmegaConf.merge(base_cfg, env_cfg)
    if "defaults" in cfg:
        cfg.pop("defaults")
    return cfg


class SpriteSyntheticGymEnv(gym.Env):
    """
    Gym-compatible wrapper over the sprite-based synthetic environments (TargetEnv, PushEnv, etc.).
    It builds the underlying env from a YAML config stem and delegates the Gym API.
    """

    metadata = {"render_modes": ["rgb_array"], "render.modes": ["rgb_array"]}

    def __init__(self, config_stem: str, env_type: str, seed: int = 0):
        cfg = _load_synthetic_env_cfg(config_stem)
        env_cls_map = {
            "TargetEnv": TargetEnv,
            "PushEnv": PushEnv,
            "MazeEnv": MazeEnv,
            "OddOneOutEnv": OddOneOutEnv,
            "RandomObjsEnv": RandomObjsEnv,
        }
        assert env_type in env_cls_map, f"Unknown env_type: {env_type}"
        self._env = env_cls_map[env_type](cfg, seed=seed)
        self.action_space = self._env.action_space
        self.observation_space = self._env.observation_space

    def seed(self, seed=None):
        return self._env.seed(seed)

    def reset(self, seed=None, **kwargs):
        return self._env.reset(seed=seed, **kwargs)

    def step(self, action):
        return self._env.step(action)

    def render(self, mode=None):
        if mode is None:
            mode = "rgb_array"
        return self._env.render(mode=mode)

    def close(self):
        self._env.close()

#obj. goal
gym.register(
    id="TargetEnv-v0",
    entry_point="zoo.ocrl.env.synthetic_envs:SpriteSyntheticGymEnv",
    kwargs={"config_stem": "target-N4C4S3S1", "env_type": "TargetEnv"},
)

#obj.inter
gym.register(
    id="PushEnv-v0",
    entry_point="zoo.ocrl.env.synthetic_envs:SpriteSyntheticGymEnv",
    kwargs={"config_stem": "push-N3C4S1S1", "env_type": "PushEnv"},
)

#obj.comp
gym.register(
    id="OddOneOutEnvObject-v0",
    entry_point="zoo.ocrl.env.synthetic_envs:SpriteSyntheticGymEnv",
    kwargs={"config_stem": "odd-one-out-N4C2S2S1-oc", "env_type": "OddOneOutEnv"},
)

#prop. comp
gym.register(
    id="OddOneOutEnvProperty-v0",
    entry_point="zoo.ocrl.env.synthetic_envs:SpriteSyntheticGymEnv",
    kwargs={"config_stem": "odd-one-out-N4C2S2S1", "env_type": "OddOneOutEnv"},
)
