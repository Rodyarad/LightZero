import cv2
import numpy as np
import torch
from omegaconf import OmegaConf

from zoo.ocrl.env.synthetic_envs import SpriteSyntheticGymEnv, _load_synthetic_env_cfg
from zoo.ocr.slate.slate import SLATE
from zoo.ocr.tools import obs_to_tensor


def _obs_to_tensor_safe(obs: np.ndarray, device: str) -> torch.Tensor:
    # Sprite env observations can have negative strides after wrappers/ops.
    obs = np.ascontiguousarray(obs)
    return obs_to_tensor(obs[np.newaxis], device=device)


if __name__ == '__main__':
    env_config_stem = 'target-N4C4S3S1'
    env_config = _load_synthetic_env_cfg(env_config_stem)
    seed = 32
    env = SpriteSyntheticGymEnv(config_stem=env_config_stem, env_type='TargetEnv', seed=seed)
    env.action_space.seed(seed)

    ocr_config_path = 'zoo/ocr/slate/config/slate_ocrl.yaml'
    config_ocr = OmegaConf.load(ocr_config_path)
    slate = SLATE(config_ocr, env_config, observation_space=None, preserve_slot_order=True)
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    slate.to(device)

    checkpoint_path = 'zoo/ocr/slate_weights/slate_ocrl.pth'
    state_dict = torch.load(checkpoint_path, map_location=device)['ocr_module_state_dict']
    slate._module.load_state_dict(state_dict)
    slate.requires_grad_(False)
    slate.eval()

    obs = env.reset()
    if isinstance(obs, tuple):
        obs = obs[0]
    obs = _obs_to_tensor_safe(obs, device=device)
    slots = [slate._module._get_slots(obs, prev_slots=None)]
    samples = [slate._module.get_samples(obs, prev_slots=None)]
    prev_slots = slots[-1]
    done = False
    while not done:
        obs, rew, done, info = env.step(env.action_space.sample())
        obs = _obs_to_tensor_safe(obs, device=device)
        slots.append(slate._module._get_slots(obs, prev_slots=prev_slots))
        samples.append(slate._module.get_samples(obs, prev_slots=prev_slots))
        prev_slots = slots[-1]

    first_sample = samples[0]
    cv2.imwrite('ocrl.png', cv2.cvtColor(first_sample['samples'][0], cv2.COLOR_RGB2BGR))
