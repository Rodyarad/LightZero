import cv2
import numpy as np
import torch

from zoo.ocrl.env.synthetic_envs import SpriteSyntheticGymEnv, _load_synthetic_env_cfg
from zoo.ocr.steve import load_steve_from_ckpt, visualize_steve_samples


def _obs_to_tensor_safe(obs: np.ndarray, device: str, image_size: int) -> torch.Tensor:
    obs = np.ascontiguousarray(obs)
    if obs.shape[0] != image_size or obs.shape[1] != image_size:
        obs = cv2.resize(obs, (image_size, image_size), interpolation=cv2.INTER_LINEAR)
    return torch.tensor(obs[np.newaxis].transpose(0, 3, 1, 2), device=device, dtype=torch.float32) / 255.0


if __name__ == '__main__':
    env_config_stem = 'target-N4C4S3S1'
    env_config = _load_synthetic_env_cfg(env_config_stem)
    seed = 7
    env = SpriteSyntheticGymEnv(config_stem=env_config_stem, env_type='TargetEnv', seed=seed)
    env.action_space.seed(seed)

    ocr_config_path = 'zoo/ocr/steve/configs/steve_ocrl.yaml'
    checkpoint_path = 'zoo/ocr/steve_weights/steve_ocrl.pt'
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    image_size = 128

    steve = load_steve_from_ckpt(
        config_path=ocr_config_path,
        checkpoint_path=checkpoint_path,
        image_size=image_size,
        image_channels=env_config.obs_channels,
        device=device,
    )
    steve.requires_grad_(False)
    steve.eval()

    obs = env.reset()
    if isinstance(obs, tuple):
        obs = obs[0]
    obs = _obs_to_tensor_safe(obs, device=device, image_size=image_size)
    slots = [steve.extract_slots(obs, prev_slots=None)]
    samples = [visualize_steve_samples(steve, obs, prev_slots=None)]
    prev_slots = slots[-1]
    done = False
    while not done:
        obs, rew, done, info = env.step(env.action_space.sample())
        obs = _obs_to_tensor_safe(obs, device=device, image_size=image_size)
        slots.append(steve.extract_slots(obs, prev_slots=prev_slots))
        samples.append(visualize_steve_samples(steve, obs, prev_slots=prev_slots))
        prev_slots = slots[-1]

    first_sample = samples[0]
    cv2.imwrite('ocrl_steve.png', cv2.cvtColor(first_sample['samples'][0], cv2.COLOR_RGB2BGR))
