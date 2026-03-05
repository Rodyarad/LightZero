import os
from collections import namedtuple

import cv2
import numpy as np
import torch
from omegaconf import OmegaConf

from zoo.ocr.slate.slate import SLATE


def load_slate(
    ocr_config_path: str,
    checkpoint_path: str,
    obs_size: int = 64,
    obs_channels: int = 3,
    device: str = "cuda",
) -> SLATE:
    """
    Load SLATE model and its decoder for a given environment configuration.
    """
    ocr_config = OmegaConf.load(ocr_config_path)
    EnvConfig = namedtuple("EnvConfig", ["obs_size", "obs_channels"])
    env_config = EnvConfig(obs_size=obs_size, obs_channels=obs_channels)

    slate = SLATE(ocr_config, env_config, observation_space=None, preserve_slot_order=True)
    state_dict = torch.load(checkpoint_path, map_location=device)["ocr_module_state_dict"]
    slate._module.load_state_dict(state_dict)
    slate.to(device)
    slate.eval()
    slate.requires_grad_(False)
    return slate


def reconstruct_from_slots(
    slots_path: str,
    output_dir: str,
    ocr_config_path: str = "zoo/ocr/slate/config/navigation5x5.yaml",
    checkpoint_path: str = "zoo/ocr/slate_weights/navigation5x5.pth",
    device: str = "cuda",
) -> None:
    """
    Read UniZero slots from .npy and decode them into images via SLATE.

    Expected slots.npy shape: (T, num_slots, slot_dim), where num_slots and slot_dim
    match the SLATE/UniZero configuration (for Navigation5x5: 6 and 64).
    """
    os.makedirs(output_dir, exist_ok=True)

    # Load slots produced by UniZero / SlotExtractorWrapper
    slots_np = np.load(slots_path)  # (T, num_slots, slot_dim)
    if slots_np.ndim != 3:
        raise ValueError(f"Expected slots of shape (T, num_slots, slot_dim), got {slots_np.shape}")

    T, num_slots, slot_dim = slots_np.shape
    print(f"Loaded slots from {slots_path} with shape {slots_np.shape}")

    slate = load_slate(
        ocr_config_path=ocr_config_path,
        checkpoint_path=checkpoint_path,
        obs_size=64,      # as in shapes2d config
        obs_channels=3,   # RGB
        device=device,
    )

    # Convert to tensor
    slots_tensor = torch.from_numpy(slots_np).to(device=device, dtype=torch.float32)  # (T, N, D)

    with torch.no_grad():
        # _gen_imgs: (B, num_slots, slot_size) -> (B, C, H, W)
        recons = slate._module._gen_imgs(slots_tensor)  # (T, 3, 64, 64)

    recons = recons.clamp(0, 1).cpu().numpy()  # [0,1], shape (T, C, H, W)

    for t in range(T):
        img_chw = recons[t]           # (C, H, W)
        img_hwc = np.transpose(img_chw, (1, 2, 0))  # (H, W, C)
        img_hwc = (img_hwc * 255.0).astype(np.uint8)

        # OpenCV expects BGR
        img_bgr = cv2.cvtColor(img_hwc, cv2.COLOR_RGB2BGR)
        out_path = os.path.join(output_dir, f"slot_recon_{t:04d}.jpg")
        ok = cv2.imwrite(out_path, img_bgr)
        if not ok:
            print(f"Warning: failed to write {out_path}")

    print(f"Saved {T} reconstructed frames to {output_dir}")


if __name__ == "__main__":
    slots_path = "visuals/unizero_slots_env0_episode001.npy"
    output_dir = "visuals/slots_episode_001_recon"

    reconstruct_from_slots(
        slots_path=slots_path,
        output_dir=output_dir,
        ocr_config_path="zoo/ocr/slate/config/navigation5x5.yaml",
        checkpoint_path="zoo/ocr/slate_weights/navigation5x5.pth",
        device="cuda",
    )

