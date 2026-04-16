import cv2
import numpy as np
import torch
from zoo.maniskill.env.maniskill3 import ManiSkill
from zoo.ocr.savi import load_savi_from_ckpt
from zoo.ocr.tools import obs_to_tensor


def tensor_image_to_bgr(image: torch.Tensor) -> np.ndarray:
    image = image.detach().cpu().clamp(0, 1)
    image = (image.permute(1, 2, 0).numpy() * 255).astype(np.uint8)
    return cv2.cvtColor(image, cv2.COLOR_RGB2BGR)


def build_slots_only_image(rgbs: torch.Tensor, masks: torch.Tensor) -> torch.Tensor:
    # Keep only per-slot renders and concatenate them horizontally.
    # Input tensors are expected as (1, num_slots, C, H, W) from SAVi decode output.
    slot_ids = masks.argmax(dim=1, keepdim=True)
    hard_masks = torch.zeros_like(masks).scatter_(1, slot_ids, 1.0)
    individual_slots = (rgbs * hard_masks)[0]  # (num_slots, C, H, W)
    return torch.cat([individual_slots[i] for i in range(individual_slots.shape[0])], dim=-1)


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

    from omegaconf import OmegaConf
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
    slot_samples = []
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
            slot_samples.append(build_slots_only_image(decoded['rgbs'].cpu()[0], decoded['masks'].cpu()[0]))
            prev_slots = current_slots

            if done:
                break

            obs, rew, done, info = env.step(env.action_space.sample())
            step_count += 1

    cv2.imwrite('maniskill_savi_slots.png', tensor_image_to_bgr(slot_samples[-1]))
