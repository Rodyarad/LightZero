import cv2
import numpy as np
import torch
import torch.nn as nn
from omegaconf import OmegaConf
from zoo.maniskill.env.maniskill3 import ManiSkill
from zoo.ocr.savi import load_savi_from_ckpt
from zoo.ocr.tools import obs_to_tensor
from zoo.ocr.savi.visualizations import make_grid


def convert_one_hot(masks: torch.Tensor) -> torch.Tensor:
    mask_argmax = torch.argmax(masks, dim=-3)
    masks_hard = nn.functional.one_hot(mask_argmax, masks.shape[-3]).to(torch.float32)
    return masks_hard.transpose(-1, -2).transpose(-2, -3)


def get_masks_hard(images: torch.Tensor, masks: torch.Tensor):
    masks = masks.detach().cpu()
    masks_hard = convert_one_hot(masks.squeeze(-3)).unsqueeze(-3)
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
    seed = 0
    image_size = 64
    max_steps = 10
    env = ManiSkill(
        reward_mode='normalized_dense',
        pose_reward_coef=0.01,
        place_reward_coef=0.1,
        image_size=image_size,
    )
    env.seed(seed)

    ocr_config_path = 'zoo/ocr/savi/configs/savi_maniskill.yaml'
    checkpoint_path = 'zoo/ocr/savi_weights/savi_maniskill_nslot-3.ckpt'
    config_ocr = OmegaConf.load(ocr_config_path)

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    savi = load_savi_from_ckpt(
        cfg=config_ocr,
        ckpt_path=checkpoint_path,
        image_size=(image_size, image_size),
        device=device,
    )
    savi.requires_grad_(False)
    savi.eval()

    slots = []
    frame_samples = []
    prev_slots = None
    obs = env.reset()
    done = False
    step_count = 0

    with torch.no_grad():
        while step_count < max_steps:
            obs_tensor = obs_to_tensor(obs[np.newaxis], device=device)
            current_slots = savi.extract_slots(obs_tensor, prev_slots=prev_slots)
            slots.append(current_slots)

            decoded = savi.decode(current_slots.unsqueeze(1))
            masks = decoded['masks'][:, 0].cpu()
            masks_hard = convert_one_hot(masks.squeeze(-3))
            frame_samples.append(grid(obs_tensor.cpu(), masks_hard))
            prev_slots = current_slots

            if done:
                break

            obs, rew, done, info = env.step(env.action_space.sample())
            step_count += 1

    frame = (frame_samples[-1] * 255).astype(np.uint8)
    cv2.imwrite('maniskill_savi_slots.png', cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))
