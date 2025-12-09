from gym.wrappers import TimeLimit
from omegaconf import OmegaConf

from cw_envs import CwTargetEnv

if __name__ == '__main__':
    env_config_path = 'cw_envs/config/reaching-hard_orig.yaml'
    env_config = OmegaConf.load(env_config_path)
    seed = 0
    env = CwTargetEnv(env_config, seed)
    env.action_space.seed(seed)
    env = TimeLimit(env, env.unwrapped._max_episode_length)

    obs = env.reset()
    done = False
    while not done:
        obs, rew, done, info = env.step(env.action_space.sample())
        print(obs.shape)
