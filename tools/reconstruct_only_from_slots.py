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
    slot_type: str,
    ocr_config_path: str = "zoo/ocr/slate/config/navigation5x5.yaml",
    checkpoint_path: str = "zoo/ocr/slate_weights/navigation5x5.pth",
    device: str = "cuda",
) -> None:
    """
    Read slots from .npy and decode them into images via SLATE.

    Args:
        slots_path: Path to the .npy file with slots of shape (T, num_slots, slot_dim).
        output_dir: Directory where reconstructed images will be saved.
        slot_type: Either "sa" (slot-attention / encoder) or "dynamics" (world-model predicted).
                   Used as a prefix for output filenames.
        ocr_config_path: Path to the SLATE config YAML.
        checkpoint_path: Path to the SLATE checkpoint.
        device: Torch device.
    """
    os.makedirs(output_dir, exist_ok=True)

    slots_np = np.load(slots_path)  # (T, num_slots, slot_dim)
    if slots_np.ndim != 3:
        raise ValueError(f"Expected slots of shape (T, num_slots, slot_dim), got {slots_np.shape}")

    T, num_slots, slot_dim = slots_np.shape
    print(f"[{slot_type}] Loaded slots from {slots_path} with shape {slots_np.shape}")

    slate = load_slate(
        ocr_config_path=ocr_config_path,
        checkpoint_path=checkpoint_path,
        obs_size=64,
        obs_channels=3,
        device=device,
    )

    slots_tensor = torch.from_numpy(slots_np).to(device=device, dtype=torch.float32)

    with torch.no_grad():

        recon_dict = slate._module.get_slotwise_reconstructions_from_slots(slots_tensor)

    combined_recons = recon_dict["combined_recons"].clamp(0, 1).cpu().numpy()
    for t in range(T):
        img_chw = combined_recons[t]  # (C, H, num_slots * W)
        img_hwc = np.transpose(img_chw, (1, 2, 0))
        img_hwc = (img_hwc * 255.0).astype(np.uint8)

        img_bgr = cv2.cvtColor(img_hwc, cv2.COLOR_RGB2BGR)
        out_path = os.path.join(output_dir, f"{slot_type}_recon_{t:04d}.jpg")
        ok = cv2.imwrite(out_path, img_bgr)
        if not ok:
            print(f"Warning: failed to write {out_path}")

    print(f"[{slot_type}] Saved {T} reconstructed frames to {output_dir}")


if __name__ == "__main__":
    ocr_config_path = "zoo/ocr/slate/config/slate_ocrl.yaml"
    checkpoint_path = "zoo/ocr/slate_weights/slate_ocrl.pth"
    # ocr_config_path = "zoo/ocr/slate/config/slate_3d.yaml"
    # checkpoint_path = "zoo/ocr/slate_weights/slate_3d.pth"
    device = "cuda"

    # --- Slot-attention (encoder) slots ---
    sa_slots_path = "visuals/sa_slots_env0_episode001.npy"
    sa_output_dir = "visuals/sa_slots_episode_001_recon"

    reconstruct_from_slots(
        slots_path=sa_slots_path,
        output_dir=sa_output_dir,
        slot_type="sa",
        ocr_config_path=ocr_config_path,
        checkpoint_path=checkpoint_path,
        device=device,
    )