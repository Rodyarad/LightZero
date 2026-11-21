"""Frozen DINOv2 features from full-size RGB images masked by own_seg_v1."""

import gym
import numpy as np
import timm
import torch
from torch import nn


class DinoObjectEncoder(nn.Module):
    """Images (H,W,3) + labels (H,W) -> embeddings (K,384).

    K masked images preserve the original canvas, coordinates and RGB values.
    Unselected pixels are zeroed before ImageNet normalization. CLS tokens
    represent the full masked image; no Slot Attention or learned projection.
    """

    def __init__(self, num_objects, device="cpu", weights_path=None):
        super().__init__()
        self.num_objects = num_objects
        self.device = torch.device(device)
        self.backbone = timm.create_model(
            "vit_small_patch14_dinov2", pretrained=weights_path is None,
            num_classes=0, dynamic_img_size=True,
        )
        if weights_path is not None:
            if str(weights_path).endswith(".safetensors"):
                from safetensors.torch import load_file
                checkpoint = load_file(str(weights_path))
            else:
                checkpoint = torch.load(weights_path, map_location="cpu")
            self.backbone.load_state_dict(checkpoint.get("state_dict", checkpoint))
        self.backbone.requires_grad_(False).eval().to(self.device)
        self.register_buffer("mean", torch.tensor((0.485, 0.456, 0.406)).view(1, 3, 1, 1))
        self.register_buffer("std", torch.tensor((0.229, 0.224, 0.225)).view(1, 3, 1, 1))
        self.to(self.device)

    @torch.inference_mode()
    def forward(self, rgb, labels):
        if rgb.ndim != 3 or rgb.shape[-1] != 3 or labels.shape != rgb.shape[:2]:
            raise ValueError(f"Expected RGB (H,W,3) and labels (H,W), got {rgb.shape}, {labels.shape}")
        if rgb.shape[0] % 14 or rgb.shape[1] % 14:
            raise ValueError("DINOv2 patch-14 requires H and W divisible by 14")
        image = torch.as_tensor(np.ascontiguousarray(rgb), device=self.device).float() / 255
        masks = torch.as_tensor(np.ascontiguousarray(labels), device=self.device)[None] == torch.arange(
            1, self.num_objects + 1, device=self.device
        )[:, None, None]
        visible = masks.flatten(1).any(1)
        masked = image.permute(2, 0, 1)[None] * masks[:, None]
        embeddings = torch.zeros((self.num_objects, self.backbone.num_features), device=self.device)
        if visible.any():
            tokens = self.backbone.forward_features((masked[visible] - self.mean) / self.std)
            embeddings[visible] = tokens[:, 0].float()
        return embeddings.cpu().numpy()


class DinoObjectWrapper(gym.Wrapper):
    """Replace image observations with (num_objects, 384) GT-object features."""

    def __init__(self, env, source_env, num_objects, device="cpu", weights_path=None):
        super().__init__(env)
        self.source_env = source_env
        self.encoder = DinoObjectEncoder(num_objects, device=device, weights_path=weights_path)
        self.observation_space = gym.spaces.Box(
            low=-np.inf, high=np.inf, shape=(num_objects, 384), dtype=np.float32
        )

    def _encode(self):
        rgb, labels = self.source_env.get_last_gt_frame()
        return self.encoder(rgb, labels)

    def reset(self, **kwargs):
        self.env.reset(**kwargs)
        return self._encode()

    def step(self, action):
        _, reward, done, info = self.env.step(action)
        return self._encode(), reward, done, info
