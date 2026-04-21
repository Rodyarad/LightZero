import math
import re

import cv2
import numpy as np
if not hasattr(np, 'bool'):
    np.bool = bool
if not hasattr(np, 'int'):
    np.int = int
if not hasattr(np, 'float'):
    np.float = float
import torch
import torch.nn as nn
import torch.nn.functional as F
from gym.wrappers import TimeLimit
from omegaconf import OmegaConf

from zoo.causal_world.env.causal_world.cw_envs import CwTargetEnv
from zoo.ocr.slotcontrast import load_from_checkpoint as load_slotcontrast_from_ckpt
from zoo.ocr.savi.visualizations import make_grid


def obs_to_tensor(obs, device):
    if len(obs.shape) == 4:
        return torch.Tensor(obs.transpose(0, 3, 1, 2)).to(device) / 255.0
    else:
        return torch.Tensor(obs).to(device)


def convert_one_hot(masks: torch.Tensor) -> torch.Tensor:
    mask_argmax = torch.argmax(masks, dim=1)
    masks_hard = nn.functional.one_hot(mask_argmax, masks.shape[1]).to(torch.float32)
    return masks_hard.permute(0, 3, 1, 2)


def get_visualization_images(images: torch.Tensor) -> torch.Tensor:
    if images.shape[1] == 3:
        return images
    return images[:, :3]


def get_masks_hard(images: torch.Tensor, masks: torch.Tensor):
    images = get_visualization_images(images)
    masks = masks.detach().cpu()
    masks_hard = convert_one_hot(masks).unsqueeze(2)
    masks_hard_vis = images.unsqueeze(1) * masks_hard + (1 - masks_hard)
    return masks_hard, masks_hard_vis


def vis(source_images: torch.Tensor, images: torch.Tensor, is_reconstruction: bool):
    if is_reconstruction:
        assert source_images.shape == images.shape, f'{source_images.shape} != {images.shape}'
        return torch.stack([source_images, images], dim=-4)

    source_images = source_images.unsqueeze(-4)
    images = images.unsqueeze(-3)
    return torch.cat([source_images, source_images * images + (1 - images)], dim=-4)


def grid(source_images: torch.Tensor, images: torch.Tensor, is_reconstruction: bool = False) -> np.ndarray:
    images = images.clamp_(0, 1)
    attention_maps = vis(source_images, images, is_reconstruction)
    log_image = attention_maps.flatten(end_dim=-4)
    log_image = make_grid(log_image, attention_maps.shape[-4], pad_color=torch.tensor([0.5, 0.5, 0.5]))
    return log_image.movedim(0, -1).cpu().numpy()


if __name__ == '__main__':
    env_config_path = 'zoo/causal_world/env/causal_world/cw_envs/config/reaching-hard_orig.yaml'
    env_config = OmegaConf.load(env_config_path)
    seed = 0
    max_steps = 3
    env = CwTargetEnv(env_config, seed)
    env.action_space.seed(seed)
    env = TimeLimit(env, env.unwrapped._max_episode_length)

    ocr_config_path = 'zoo/ocr/slotcontrast/configs/cw-reaching-hard.yaml'
    checkpoint_path = 'zoo/ocr/slotcontrast_weights/slotcontrast_cw.ckpt'
    ocr_config = OmegaConf.load(ocr_config_path)

    num_patches = int(ocr_config.globals.NUM_PATCHES)
    patch_grid = int(math.sqrt(num_patches))
    assert patch_grid * patch_grid == num_patches, f'NUM_PATCHES={num_patches} must be a square'
    model_name = str(ocr_config.globals.DINO_MODEL)
    patch_match = re.search(r'patch(\d+)', model_name)
    patch_size = int(patch_match.group(1)) if patch_match else 14
    model_input_size = patch_grid * patch_size

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    slotcontrast = load_slotcontrast_from_ckpt(
        config_path=ocr_config_path,
        checkpoint_path=checkpoint_path,
        device=device,
    )
    slotcontrast.requires_grad_(False)
    slotcontrast.eval()

    frame_samples = []
    prev_slots = None
    obs = env.reset()
    done = False
    step_count = 0

    with torch.no_grad():
        while step_count < max_steps:
            obs_tensor = obs_to_tensor(obs[np.newaxis], device=device)
            if obs_tensor.shape[-2:] != (model_input_size, model_input_size):
                model_input = F.interpolate(
                    obs_tensor,
                    size=(model_input_size, model_input_size),
                    mode='bilinear',
                    align_corners=False,
                )
            else:
                model_input = obs_tensor

            current_slots = slotcontrast.extract_slots(model_input, prev_slots=prev_slots)

            decoded = slotcontrast.decoder(current_slots.unsqueeze(1))
            masks = decoded["masks"][:, 0]
            if masks.ndim == 3:
                patch_side = int(math.sqrt(masks.shape[-1]))
                masks = masks.view(masks.shape[0], masks.shape[1], patch_side, patch_side)
            masks = F.interpolate(masks, size=obs_tensor.shape[-2:], mode='bilinear', align_corners=False).cpu()

            vis_images = get_visualization_images(obs_tensor.cpu())
            masks_hard, _ = get_masks_hard(vis_images, masks)
            frame_samples.append(grid(vis_images, masks_hard.squeeze(2)))
            prev_slots = current_slots

            if done:
                break

            obs, rew, done, info = env.step(env.action_space.sample())
            step_count += 1

    frame = (frame_samples[-1] * 255).astype(np.uint8)
    cv2.imwrite('causal_world_slotcontrast_slots.png', cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))
