from collections import namedtuple
from typing import Any, Dict, Optional

import numpy as np
import torch

SlateEnvConfig = namedtuple('SlateEnvConfig', ['obs_size', 'obs_channels'])


def build_finetune_slate(
        ocr_config_path: str,
        checkpoint_path: Optional[str],
        obs_size: int,
        obs_channels: int,
        device: str,
) -> "SLATE":
    from omegaconf import OmegaConf
    from zoo.ocr.slate.slate import SLATE

    config_ocr = OmegaConf.load(ocr_config_path)
    config_env = SlateEnvConfig(obs_size, obs_channels)
    slate = SLATE(config_ocr, config_env, observation_space=None)
    if checkpoint_path is not None:
        checkpoint = torch.load(checkpoint_path, map_location='cpu')
        slate.load(checkpoint)
    slate.to(device)
    for param in slate._module.parameters():
        if param.dtype.is_floating_point:
            param.requires_grad_(True)
    slate.train()
    return slate


def prepare_slate_obs_batch(
        raw_obs_batch: np.ndarray,
        mask_batch: np.ndarray,
        device: str,
        max_images: Optional[int] = None,
) -> torch.Tensor:
    assert raw_obs_batch.ndim == 5, f"Expected (B, T, H, W, C) raw obs, got {raw_obs_batch.shape}"
    B, T = raw_obs_batch.shape[:2]
    mask = np.asarray(mask_batch)[:, :T].astype(bool)
    images = raw_obs_batch.reshape(B * T, *raw_obs_batch.shape[2:])[mask.reshape(-1)]

    if max_images is not None and images.shape[0] > max_images:
        idx = np.random.choice(images.shape[0], size=max_images, replace=False)
        images = images[idx]

    images = torch.from_numpy(np.ascontiguousarray(images)).to(device)
    images = images.permute(0, 3, 1, 2).float() / 255.0
    return images


def raw_images_to_tensor(images: np.ndarray, device: str) -> torch.Tensor:
    x = torch.from_numpy(np.ascontiguousarray(images)).to(device)
    x = torch.movedim(x, -1, -3).float() / 255.0
    return x


@torch.no_grad()
def encode_images_to_slots(
        slate: "SLATE",
        images: np.ndarray,
        device: str,
        chunk_size: int = 256,
) -> torch.Tensor:
    module = slate._module
    was_training = module.training
    module.eval()
    try:
        outputs = []
        for beg in range(0, images.shape[0], chunk_size):
            frames = raw_images_to_tensor(images[beg:beg + chunk_size], device)
            init_slots = module._get_slots(frames, prev_slots=None)
            outputs.append(module._get_slots(frames, prev_slots=init_slots))
        slots = torch.cat(outputs, dim=0)
    finally:
        if was_training:
            module.train()
    return slots


@torch.no_grad()
def encode_image_sequences_to_slots(
        slate: "SLATE",
        images: np.ndarray,
        device: str,
) -> torch.Tensor:
    assert images.ndim == 5, f"Expected (B, T, H, W, C) raw obs, got {images.shape}"
    module = slate._module
    was_training = module.training
    module.eval()
    try:
        frames = raw_images_to_tensor(images, device)
        prev_slots = None
        outputs = []
        for t in range(frames.shape[1]):
            frame = frames[:, t]
            if prev_slots is None:
                prev_slots = module._get_slots(frame, prev_slots=None)
            prev_slots = module._get_slots(frame, prev_slots=prev_slots)
            outputs.append(prev_slots)
        slots = torch.stack(outputs, dim=1)
    finally:
        if was_training:
            module.train()
    return slots


def encode_flat_windows_to_slots(
        slate: "SLATE",
        images: np.ndarray,
        window_len: int,
        device: str,
) -> torch.Tensor:
    assert images.ndim == 4, f"Expected flat (N, H, W, C) raw obs, got {images.shape}"
    assert images.shape[0] % window_len == 0, \
        f"Flat batch of {images.shape[0]} frames is not divisible into windows of {window_len}"
    windows = images.reshape(-1, window_len, *images.shape[1:])
    slots = encode_image_sequences_to_slots(slate, windows, device)
    return slots.reshape(-1, slots.shape[-2], slots.shape[-1])


class CollectorSlotEncoder:
    def __init__(self, slate: "SLATE", device: str) -> None:
        assert slate is not None, \
            "CollectorSlotEncoder requires the policy's SLATE model (finetune_slate=True); " \
            "note that random-collect policies do not provide one."
        self._slate = slate
        self._device = device
        self._last_slots = {}

    def has_state(self, env_id: int) -> bool:
        return env_id in self._last_slots

    def reset(self, env_id: Optional[int] = None) -> None:
        if env_id is None:
            self._last_slots.clear()
        else:
            self._last_slots.pop(env_id, None)

    def encode(self, frames_dict: Dict[int, np.ndarray]) -> Dict[int, np.ndarray]:
        if not frames_dict:
            return {}
        env_ids = sorted(frames_dict)
        frames = np.stack([np.asarray(frames_dict[env_id]) for env_id in env_ids])
        module = self._slate._module
        was_training = module.training
        module.eval()
        try:
            with torch.no_grad():
                x = raw_images_to_tensor(frames, self._device)
                fresh_positions = [k for k, env_id in enumerate(env_ids) if env_id not in self._last_slots]
                if fresh_positions:
                    init_slots = module._get_slots(x[fresh_positions], prev_slots=None)
                    for k, position in enumerate(fresh_positions):
                        self._last_slots[env_ids[position]] = init_slots[k]
                prev_slots = torch.stack([self._last_slots[env_id] for env_id in env_ids])
                slots = module._get_slots(x, prev_slots=prev_slots)
                for k, env_id in enumerate(env_ids):
                    self._last_slots[env_id] = slots[k]
        finally:
            if was_training:
                module.train()
        return {env_id: slots[k].cpu().numpy() for k, env_id in enumerate(env_ids)}

    def encode_init(self, frames_dict: Dict[int, np.ndarray]) -> Dict[int, np.ndarray]:
        slots_dict = {
            env_id: self._last_slots[env_id].cpu().numpy()
            for env_id in frames_dict if env_id in self._last_slots
        }
        fresh_frames = {env_id: frame for env_id, frame in frames_dict.items() if env_id not in self._last_slots}
        slots_dict.update(self.encode(fresh_frames))
        return slots_dict


def slate_finetune_step(
        slate: "SLATE",
        images: torch.Tensor,
        step: int,
) -> Dict[str, Any]:
    slate.train()
    metrics = slate.update(images, None, step)
    log = {}
    for key, value in metrics.items():
        if isinstance(value, torch.Tensor):
            value = value.detach().cpu()
            value = value.item() if value.numel() == 1 else value.mean().item()
        log[f'slate/{key}'] = float(value)
    log['slate/num_images'] = float(images.shape[0])
    log['slate/step'] = float(step)
    return log
