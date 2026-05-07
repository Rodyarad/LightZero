import argparse
import os
from collections import namedtuple
from pathlib import Path
from typing import Dict, List

import cv2
import numpy as np
import torch
from easydict import EasyDict
from omegaconf import OmegaConf

from lzero.model.sampled_unizero_model import SampledUniZeroModel
from lzero.model.unizero_model import UniZeroModel

ENV_DEFAULTS = {
    "ocrl": {
        "ocr_backend": "slate",
        "dinosaur_preset": "robosuite",
        "ocr_config_path": "zoo/ocr/slate/config/slate_ocrl.yaml",
        "ocr_checkpoint_path": "zoo/ocr/slate_weights/slate_ocrl.pth",
        "obs_path": "visuals/random_policy_log/random_obs.npy",
        "slots_path": "visuals/random_policy_log/random_slots.npy",
        "actions_path": "visuals/random_policy_log/random_actions.npy",
        "action_space_size": 4,
        "num_slots": 6,
        "slot_dim": 192,
        "num_unroll_steps": 10,
        "infer_context_length": 4,
        "game_segment_length": 20,
        "num_simulations": 50,
        "support_size": 601,
        "continuous_action_space": False,
        "num_of_sampled_actions": None,
        "policy_entropy_weight": 5e-3,
        "obs_size": 64,
    },
    "causal_world": {
        "ocr_backend": "slate",
        "dinosaur_preset": "robosuite",
        "ocr_config_path": "zoo/ocr/slate/config/slate_3d.yaml",
        "ocr_checkpoint_path": "zoo/ocr/slate_weights/slate_3d.pth",
        "obs_path": "visuals/random_policy_log/random_obs.npy",
        "slots_path": "visuals/random_policy_log/random_slots.npy",
        "actions_path": "visuals/random_policy_log/random_actions.npy",
        "action_space_size": 3,
        "num_slots": 10,
        "slot_dim": 192,
        "num_unroll_steps": 5,
        "infer_context_length": 2,
        "game_segment_length": 100,
        "num_simulations": 50,
        "support_size": 101,
        "continuous_action_space": True,
        "num_of_sampled_actions": 20,
        "policy_entropy_weight": 5e-2,
        "obs_size": 64,
    },
    "robosuite": {
        "ocr_backend": "dinosaur",
        "dinosaur_preset": "robosuite",
        "ocr_config_path": "",
        "ocr_checkpoint_path": "zoo/ocr/dinosaur_weights/robosuite.ckpt",
        "obs_path": "visuals/random_policy_log/random_obs.npy",
        "slots_path": "visuals/random_policy_log/random_slots.npy",
        "actions_path": "visuals/random_policy_log/random_actions.npy",
        "action_space_size": 4,
        "num_slots": 5,
        "slot_dim": 64,
        "num_unroll_steps": 5,
        "infer_context_length": 2,
        "game_segment_length": 100,
        "num_simulations": 50,
        "support_size": 101,
        "continuous_action_space": True,
        "num_of_sampled_actions": 20,
        "policy_entropy_weight": 5e-2,
        "obs_size": 224,
    },
    "maniskill": {
        "ocr_backend": "dinosaur",
        "dinosaur_preset": "maniskill",
        "ocr_config_path": "",
        "ocr_checkpoint_path": "zoo/ocr/dinosaur_weights/maniskill.ckpt",
        "obs_path": "visuals/random_policy_log/random_obs.npy",
        "slots_path": "visuals/random_policy_log/random_slots.npy",
        "actions_path": "visuals/random_policy_log/random_actions.npy",
        "action_space_size": 8,
        "num_slots": 3,
        "slot_dim": 64,
        "num_unroll_steps": 5,
        "infer_context_length": 2,
        "game_segment_length": 100,
        "num_simulations": 50,
        "support_size": 101,
        "continuous_action_space": True,
        "num_of_sampled_actions": 20,
        "policy_entropy_weight": 5e-2,
        "obs_size": 224,
    },
    "maniskill_slotcontrast": {
        "ocr_backend": "slotcontrast",
        "dinosaur_preset": "maniskill",
        "ocr_config_path": "",
        "ocr_checkpoint_path": "zoo/ocr/slotcontrast_weights/slotcontrast_maniskill.ckpt",
        "slotcontrast_config_path": "zoo/ocr/slotcontrast/configs/slotcontrast_maniskill.yaml",
        "obs_path": "visuals/random_policy_log/random_obs.npy",
        "slots_path": "visuals/random_policy_log/random_slots.npy",
        "actions_path": "visuals/random_policy_log/random_actions.npy",
        "action_space_size": 8,
        "num_slots": 3,
        "slot_dim": 64,
        "num_unroll_steps": 5,
        "infer_context_length": 2,
        "game_segment_length": 100,
        "num_simulations": 50,
        "support_size": 101,
        "continuous_action_space": True,
        "num_of_sampled_actions": 20,
        "policy_entropy_weight": 5e-2,
        "obs_size": 336,
    },
    "vizdoom_slotcontrast": {
        "ocr_backend": "slotcontrast",
        "dinosaur_preset": "robosuite",
        "ocr_config_path": "",
        "ocr_checkpoint_path": "zoo/ocr/slotcontrast_weights/slotcontrast_vizdoom.ckpt",
        "slotcontrast_config_path": "zoo/ocr/slotcontrast/configs/vizdoom_sc.yaml",
        "obs_path": "visuals/random_policy_log/random_obs.npy",
        "slots_path": "visuals/random_policy_log/random_slots.npy",
        "actions_path": "visuals/random_policy_log/random_actions.npy",
        "action_space_size": 4,
        "num_slots": 7,
        "slot_dim": 64,
        "num_unroll_steps": 10,
        "infer_context_length": 4,
        "game_segment_length": 20,
        "num_simulations": 50,
        "support_size": 601,
        "continuous_action_space": False,
        "num_of_sampled_actions": None,
        "policy_entropy_weight": 5e-3,
        "obs_size": 336,
    },
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Visualize GT vs dynamics slots for random-policy rollout."
    )
    parser.add_argument(
        "--policy-version",
        type=str,
        default="discrete",
        choices=["discrete", "sampled"],
        help="Model family: discrete UniZero or sampled UniZero.",
    )
    parser.add_argument(
        "--env-type",
        type=str,
        default="ocrl",
        choices=["ocrl", "causal_world", "robosuite", "maniskill", "maniskill_slotcontrast", "vizdoom_slotcontrast"],
        help="Environment preset to select model/OCR/default-path config.",
    )
    parser.add_argument("--model-path", type=str, default="oc_agents_weights/oz_stica_cw_slate_seed7.pth.tar")
    parser.add_argument("--obs-path", type=str, default="")
    parser.add_argument("--slots-path", type=str, default="")
    parser.add_argument("--actions-path", type=str, default="")
    parser.add_argument(
        "--obs-size",
        type=int,
        default=None,
        help="Resize GT obs to square size (default from --env-type preset).",
    )
    parser.add_argument(
        "--frame-index-offset",
        type=int,
        default=-1,
        help="obs index for step t is t+offset. -1 means auto.",
    )
    parser.add_argument("--start-step", type=int, default=0)
    parser.add_argument("--output-dir", type=str, default="visuals/random_policy_pred_rollout_samples")
    parser.add_argument("--step-filename-template", type=str, default="random_step_{step:04d}.jpg")
    parser.add_argument(
        "--merge-steps",
        type=int,
        nargs="+",
        default=[0,22,47],
        help="Optional list of step indices to merge into one image.",
    )
    parser.add_argument("--merged-output-name", type=str, default="random_selected_steps_overview.jpg")
    parser.add_argument(
        "--highlight-slots",
        type=int,
        nargs="+",
        default=[],
        help="Slot indices to highlight with colored borders.",
    )
    parser.add_argument(
        "--highlight-colors",
        type=int,
        nargs="+",
        default=None,
        help="Per-highlight slot color codes aligned with --highlight-slots: 1=red, 0=blue, 2=green.",
    )
    parser.add_argument("--ocr-config-path", type=str, default="")
    parser.add_argument("--ocr-checkpoint-path", type=str, default="")
    parser.add_argument("--slotcontrast-config-path", type=str, default="")
    parser.add_argument(
        "--dinosaur-mask-mode",
        type=str,
        default="auto",
        choices=("auto", "soft", "hard"),
        help=(
            "Dinosaur only: soft = decoder masks; hard = one winner slot per pixel (argmax); "
            "auto = hard for maniskill, soft otherwise."
        ),
    )
    parser.add_argument("--device", type=str, default="cuda")
    args = parser.parse_args()
    env_defaults = ENV_DEFAULTS[args.env_type]
    if not args.obs_path:
        args.obs_path = env_defaults["obs_path"]
    if not args.slots_path:
        args.slots_path = env_defaults["slots_path"]
    if not args.actions_path:
        args.actions_path = env_defaults["actions_path"]
    args.ocr_backend = str(env_defaults["ocr_backend"])
    args.dinosaur_preset = str(env_defaults["dinosaur_preset"])
    if env_defaults["ocr_backend"] == "slate":
        if not args.ocr_config_path:
            args.ocr_config_path = env_defaults["ocr_config_path"]
    elif env_defaults["ocr_backend"] == "slotcontrast":
        if not args.slotcontrast_config_path:
            args.slotcontrast_config_path = env_defaults["slotcontrast_config_path"]
    if not args.ocr_checkpoint_path:
        args.ocr_checkpoint_path = env_defaults["ocr_checkpoint_path"]
    if args.obs_size is None:
        args.obs_size = int(env_defaults["obs_size"])
    if args.ocr_backend == "dinosaur":
        if args.dinosaur_mask_mode == "hard":
            args.dinosaur_hard_masks = True
        elif args.dinosaur_mask_mode == "soft":
            args.dinosaur_hard_masks = False
        else:
            args.dinosaur_hard_masks = args.env_type == "maniskill"
    else:
        args.dinosaur_hard_masks = False
    if args.highlight_colors is None:
        args.highlight_colors = [1] * len(args.highlight_slots)
    if len(args.highlight_colors) != len(args.highlight_slots):
        raise ValueError(
            "--highlight-colors must have the same length as --highlight-slots. "
            f"Got {len(args.highlight_colors)} vs {len(args.highlight_slots)}."
        )
    valid_color_codes = {0, 1, 2}
    invalid_codes = sorted({code for code in args.highlight_colors if code not in valid_color_codes})
    if invalid_codes:
        raise ValueError(
            f"Unsupported highlight color code(s): {invalid_codes}. "
            "Allowed values are 0 (blue), 1 (red), 2 (green)."
        )
    color_code_to_bgr = {0: (255, 0, 0), 1: (0, 0, 255), 2: (0, 170, 0)}
    args.highlight_color_map = {
        slot_idx: color_code_to_bgr[color_code]
        for slot_idx, color_code in zip(args.highlight_slots, args.highlight_colors)
    }
    return args


def load_slate(
    ocr_config_path: str,
    checkpoint_path: str,
    obs_size: int,
    obs_channels: int,
    device: str,
):
    from zoo.ocr.slate.slate import SLATE

    ocr_config = OmegaConf.load(ocr_config_path)
    env_config = namedtuple("EnvConfig", ["obs_size", "obs_channels"])(obs_size=obs_size, obs_channels=obs_channels)

    slate = SLATE(ocr_config, env_config, observation_space=None, preserve_slot_order=True)
    state_dict = torch.load(checkpoint_path, map_location=device)["ocr_module_state_dict"]
    slate._module.load_state_dict(state_dict)
    slate.to(device)
    slate.eval()
    slate.requires_grad_(False)
    return slate


def load_dinosaur_from_checkpoint(checkpoint_path: str, device: str, preset: str):
    # Lazy import to avoid dinosaur/timm imports when backend is slotcontrast.
    from zoo.ocr.tools import load_dinosaur_from_checkpoint as _load_dinosaur_from_checkpoint

    return _load_dinosaur_from_checkpoint(checkpoint_path=checkpoint_path, device=device, preset=preset)


def load_slotcontrast_for_visualization(config_path: str, checkpoint_path: str, device: str):
    # Lazy import to avoid pulling dinosaur dependencies.
    from zoo.ocr.slotcontrast import load_from_checkpoint as load_slotcontrast_from_checkpoint

    model = load_slotcontrast_from_checkpoint(
        config_path=config_path,
        checkpoint_path=checkpoint_path,
        device=device,
    )
    model.eval()
    model.requires_grad_(False)
    return model


def obs_to_tensor(obs, device):
    if len(obs.shape) == 4:
        return torch.Tensor(obs.transpose(0, 3, 1, 2)).to(device) / 255.0
    return torch.Tensor(obs).to(device)


def load_unizero_model(model_path: str, device: str, policy_version: str, env_type: str):
    env_defaults = ENV_DEFAULTS[env_type]
    action_space_size = int(env_defaults["action_space_size"])
    num_slots = int(env_defaults["num_slots"])
    slot_dim = int(env_defaults["slot_dim"])
    num_unroll_steps = int(env_defaults["num_unroll_steps"])
    infer_context_length = int(env_defaults["infer_context_length"])
    tokens_per_block = num_slots * 2

    world_model_cfg = EasyDict(
        dict(
            model_type="slot",
            encoder_type="resnet",
            continuous_action_space=bool(env_defaults["continuous_action_space"]),
            tokens_per_block=tokens_per_block,
            max_blocks=num_unroll_steps,
            max_tokens=tokens_per_block * num_unroll_steps,
            context_length=tokens_per_block * infer_context_length,
            gru_gating=False,
            device=device,
            analysis_sim_norm=False,
            analysis_dormant_ratio_weight_rank=False,
            action_space_size=action_space_size,
            group_size=8,
            attention="causal",
            num_layers=2,
            num_heads=8,
            embed_dim=slot_dim,
            embed_pdrop=0.1,
            resid_pdrop=0.1,
            attn_pdrop=0.1,
            support_size=int(env_defaults["support_size"]),
            max_cache_size=5000,
            env_num=1,
            latent_recon_loss_weight=0.0,
            perceptual_loss_weight=0.0,
            policy_entropy_weight=float(env_defaults["policy_entropy_weight"]),
            final_norm_option_in_head="LayerNorm",
            final_norm_option_in_encoder="LayerNorm",
            predict_latent_loss_type="mse",
            obs_type="slot",
            gamma=1.0,
            dormant_threshold=0.025,
            rotary_emb=False,
            rope_theta=10000,
            max_seq_len=8192,
            lora_r=0,
            lora_alpha=1,
            lora_dropout=0.0,
            decode_loss_mode=None,
            task_embed_option=None,
            use_task_embed=False,
            task_embed_dim=96,
            use_normal_head=True,
            use_softmoe_head=False,
            use_moe_head=False,
            num_experts_in_moe_head=4,
            moe_in_transformer=False,
            multiplication_moe_in_transformer=False,
            n_shared_experts=1,
            num_experts_per_tok=1,
            num_experts_of_moe_in_transformer=8,
            use_priority=True,
            policy_loss_type="kl",
            num_unroll_steps=num_unroll_steps,
            game_segment_length=int(env_defaults["game_segment_length"]),
            num_simulations=int(env_defaults["num_simulations"]),
            sigma_type="conditioned",
            fixed_sigma_value=0.5,
            bound_type=None,
        )
    )

    if policy_version == "sampled":
        if env_defaults["num_of_sampled_actions"] is not None:
            world_model_cfg.num_of_sampled_actions = int(env_defaults["num_of_sampled_actions"])
        model = SampledUniZeroModel(
            observation_shape=(num_slots, slot_dim),
            model_type="slot",
            action_space_size=action_space_size,
            continuous_action_space=True,
            num_of_sampled_actions=int(env_defaults["num_of_sampled_actions"]),
            norm_type="LN",
            num_res_blocks=2,
            num_channels=128,
            world_model_cfg=world_model_cfg,
        )
    else:
        model = UniZeroModel(
            observation_shape=(num_slots, slot_dim),
            model_type="slot",
            action_space_size=action_space_size,
            reward_support_range=(-300.0, 301.0, 1.0),
            value_support_range=(-300.0, 301.0, 1.0),
            norm_type="LN",
            num_res_blocks=2,
            num_channels=128,
            continuous_action_space=False,
            world_model_cfg=world_model_cfg,
        )
    model.to(device)
    model.eval()

    checkpoint = torch.load(model_path, map_location=device)
    state_dict = checkpoint["model"] if isinstance(checkpoint, dict) and "model" in checkpoint else checkpoint
    if not isinstance(state_dict, dict):
        raise RuntimeError(f"Unexpected checkpoint format at {model_path}: state_dict is not a dict.")

    normalized_state_dict: Dict[str, torch.Tensor] = {}
    for key, value in state_dict.items():
        if key.startswith("_orig_mod."):
            normalized_state_dict[key[len("_orig_mod."):]] = value
        else:
            normalized_state_dict[key] = value

    model.load_state_dict(normalized_state_dict, strict=True)
    return model


def infer_frame_offset(num_actions: int, num_obs: int, arg_offset: int) -> int:
    if arg_offset >= 0:
        return arg_offset
    if num_obs == num_actions + 1:
        return 1
    return 0


def prep_obs(obs: np.ndarray, obs_size: int) -> np.ndarray:
    if obs.shape[0] != obs_size or obs.shape[1] != obs_size:
        obs = cv2.resize(obs, (obs_size, obs_size), interpolation=cv2.INTER_NEAREST)
    if obs.dtype != np.uint8:
        if np.issubdtype(obs.dtype, np.floating):
            max_v = float(np.max(obs)) if obs.size > 0 else 1.0
            if max_v <= 1.0:
                obs = (np.clip(obs, 0.0, 1.0) * 255.0).astype(np.uint8)
            else:
                obs = np.clip(obs, 0.0, 255.0).astype(np.uint8)
        else:
            obs = np.clip(obs, 0, 255).astype(np.uint8)
    return np.ascontiguousarray(obs)


def render_slot_strip(
    ocr_model,
    obs_rgb: np.ndarray,
    slots_1x: torch.Tensor,
    device: str,
    ocr_backend: str,
    dinosaur_hard_masks: bool,
) -> np.ndarray:
    obs = obs_to_tensor(obs_rgb[np.newaxis, ...], device=device)
    if ocr_backend == "slate":
        out = ocr_model._module.get_samples(obs, prev_slots=slots_1x)
    elif ocr_backend == "slotcontrast":
        out = ocr_model.get_samples(obs, prev_slots=slots_1x)
    else:
        out = ocr_model.get_samples(obs, prev_slots=slots_1x, hard_masks=dinosaur_hard_masks)
    return out["samples"][0]


def build_step_panel(
    step: int,
    gt_strip: np.ndarray,
    dyn_strip: np.ndarray,
    obs_rgb: np.ndarray,
    tile_h: int,
    strip_w: int,
    num_slots: int,
    highlight_slots: List[int],
    highlight_color_map: Dict[int, tuple],
    resize_rasters: bool = True,
) -> np.ndarray:
    """All raster blocks match model obs: obs is tile_h×tile_h, strips are strip_w×tile_h."""
    if resize_rasters:
        gt_strip = cv2.resize(gt_strip, (strip_w, tile_h), interpolation=cv2.INTER_NEAREST)
        dyn_strip = cv2.resize(dyn_strip, (strip_w, tile_h), interpolation=cv2.INTER_NEAREST)
        obs_tile_rgb = cv2.resize(obs_rgb, (tile_h, tile_h), interpolation=cv2.INTER_NEAREST)
    else:
        obs_tile_rgb = obs_rgb

    row_gap = 8
    side_pad = 6
    left_label_w = 32
    step_text = f"Step {step + 1}"
    step_font_scale = 0.62
    step_thickness = 1
    (_, text_h), text_baseline = cv2.getTextSize(
        step_text, cv2.FONT_HERSHEY_SIMPLEX, step_font_scale, step_thickness
    )
    # Keep extra headroom so Step label never overlaps image rows.
    title_h = text_h + text_baseline + 8
    h = tile_h
    w = strip_w
    row_content_w = h + side_pad + w
    panel_h = title_h + side_pad + h + row_gap + h + side_pad
    panel_w = left_label_w + side_pad + row_content_w + side_pad
    canvas = np.full((panel_h, panel_w, 3), 255, dtype=np.uint8)

    # Step label above the two comparison rows.
    (tw, _), _ = cv2.getTextSize(step_text, cv2.FONT_HERSHEY_SIMPLEX, step_font_scale, step_thickness)
    x_obs = left_label_w + side_pad
    x_strip = x_obs + h + side_pad
    tx = x_obs + (row_content_w - tw) // 2
    ty = text_h + 3
    cv2.putText(
        canvas,
        step_text,
        (tx, ty),
        cv2.FONT_HERSHEY_SIMPLEX,
        step_font_scale,
        (25, 25, 25),
        step_thickness,
        cv2.LINE_AA,
    )

    y0 = title_h + side_pad
    y1 = y0 + h + row_gap
    obs_bgr = cv2.cvtColor(obs_tile_rgb, cv2.COLOR_RGB2BGR)
    canvas[y0 : y0 + h, x_obs : x_obs + h] = obs_bgr
    canvas[y0 : y0 + h, x_strip : x_strip + w] = cv2.cvtColor(gt_strip, cv2.COLOR_RGB2BGR)
    canvas[y1 : y1 + h, x_strip : x_strip + w] = cv2.cvtColor(dyn_strip, cv2.COLOR_RGB2BGR)

    if num_slots > 0 and len(highlight_slots) > 0:
        slot_edges = np.linspace(0, w, num_slots + 1, dtype=np.int32)
        for slot_idx in highlight_slots:
            if slot_idx < 0 or slot_idx >= num_slots:
                continue
            x0_slot = int(x_strip + slot_edges[slot_idx])
            x1_slot = int(x_strip + slot_edges[slot_idx + 1] - 1)
            if x1_slot <= x0_slot:
                continue
            border_color = highlight_color_map.get(slot_idx, (0, 0, 255))
            cv2.rectangle(canvas, (x0_slot, y0), (x1_slot, y0 + h - 1), border_color, 2)
            cv2.rectangle(canvas, (x0_slot, y1), (x1_slot, y1 + h - 1), border_color, 2)

    # Left labels as vertical words.
    gt_y_center = y0 + h // 2
    dyn_y_center = y1 + h // 2

    def _put_rotated_word(word: str, y_center: int) -> None:
        font_scale = 0.62
        thickness = 1
        (tw, th), baseline = cv2.getTextSize(word, cv2.FONT_HERSHEY_SIMPLEX, font_scale, thickness)
        patch_h = max(10, th + baseline + 4)
        patch_w = max(10, tw + 4)
        patch = np.full((patch_h, patch_w, 3), 255, dtype=np.uint8)
        cv2.putText(
            patch,
            word,
            (2, patch_h - baseline - 2),
            cv2.FONT_HERSHEY_SIMPLEX,
            font_scale,
            (30, 30, 30),
            thickness,
            cv2.LINE_AA,
        )
        # Rotate full word (not character-by-character) to vertical orientation.
        rot = cv2.rotate(patch, cv2.ROTATE_90_COUNTERCLOCKWISE)
        rh, rw = rot.shape[:2]

        y0_rot = int(y_center - rh / 2)
        y0_rot = max(0, min(y0_rot, canvas.shape[0] - rh))
        # Place label close to the image block.
        x0_rot = max(0, x_obs - rw - 2)

        roi = canvas[y0_rot : y0_rot + rh, x0_rot : x0_rot + rw]
        mask = np.any(rot < 245, axis=2)
        roi[mask] = rot[mask]

    _put_rotated_word("True", gt_y_center)
    _put_rotated_word("Model", dyn_y_center)
    return canvas


def save_merged_selected_steps(
    panels_by_step: Dict[int, np.ndarray],
    selected_steps: List[int],
    out_path: Path,
) -> None:
    chosen = [step for step in selected_steps if step in panels_by_step]
    if len(chosen) == 0:
        raise ValueError("None of selected steps are available for merging.")
    panel_w = panels_by_step[chosen[0]].shape[1]
    gap = 25
    total_h = sum(panels_by_step[s].shape[0] for s in chosen) + gap * (len(chosen) - 1)
    merged = np.full((total_h, panel_w, 3), 255, dtype=np.uint8)
    y = 0
    for i, s in enumerate(chosen):
        p = panels_by_step[s]
        merged[y : y + p.shape[0], : p.shape[1]] = p
        y += p.shape[0]
        if i < len(chosen) - 1:
            y += gap
    out_path.parent.mkdir(parents=True, exist_ok=True)
    ok = cv2.imwrite(str(out_path), merged)
    if not ok:
        raise RuntimeError(f"Failed to write merged image: {out_path}")


def main() -> None:
    args = parse_args()
    env_defaults = ENV_DEFAULTS[args.env_type]
    if env_defaults["continuous_action_space"] and args.policy_version == "discrete":
        raise ValueError(
            f"env-type={args.env_type} uses continuous actions; use --policy-version sampled."
        )
    device = args.device if torch.cuda.is_available() and args.device.startswith("cuda") else "cpu"
    os.makedirs(args.output_dir, exist_ok=True)

    obs_np = np.load(args.obs_path)
    slots_np = np.load(args.slots_path)
    actions_np = np.load(args.actions_path)

    if obs_np.ndim != 4:
        raise ValueError(f"Expected obs shape (T, H, W, C), got {obs_np.shape}")
    if slots_np.ndim != 3:
        raise ValueError(f"Expected slots shape (T, num_slots, slot_dim), got {slots_np.shape}")
    if args.policy_version == "sampled":
        if actions_np.ndim != 2:
            raise ValueError(
                f"Expected sampled continuous actions shape (T-1, action_dim), got {actions_np.shape}"
            )
    else:
        if actions_np.ndim != 1:
            raise ValueError(f"Expected actions shape (T-1,), got {actions_np.shape}")
        if not np.issubdtype(actions_np.dtype, np.integer):
            raise ValueError("Only discrete actions are supported in this script.")

    start_step = int(args.start_step)
    if start_step < 0 or start_step >= len(slots_np):
        raise ValueError(f"start-step={start_step} out of range for slots length={len(slots_np)}")

    frame_offset = infer_frame_offset(num_actions=len(actions_np), num_obs=len(obs_np), arg_offset=args.frame_index_offset)
    rollout_end_exclusive = min(len(actions_np), len(obs_np) - frame_offset)
    if start_step >= rollout_end_exclusive:
        raise ValueError(
            f"start-step={start_step} leaves no rollout. "
            f"Computed rollout_end_exclusive={rollout_end_exclusive}, frame_offset={frame_offset}."
        )

    unizero = load_unizero_model(
        args.model_path,
        device=device,
        policy_version=args.policy_version,
        env_type=args.env_type,
    )
    if args.ocr_backend == "slate":
        ocr_model = load_slate(
            ocr_config_path=args.ocr_config_path,
            checkpoint_path=args.ocr_checkpoint_path,
            obs_size=args.obs_size,
            obs_channels=3,
            device=device,
        )
    elif args.ocr_backend == "slotcontrast":
        ocr_model = load_slotcontrast_for_visualization(
            config_path=args.slotcontrast_config_path,
            checkpoint_path=args.ocr_checkpoint_path,
            device=device,
        )
    else:
        ocr_model = load_dinosaur_from_checkpoint(
            checkpoint_path=args.ocr_checkpoint_path,
            device=device,
            preset=args.dinosaur_preset,
        )

    pred_slots = torch.from_numpy(slots_np[start_step:start_step + 1]).to(device=device, dtype=torch.float32)
    if not hasattr(unizero.world_model, "latent_state"):
        unizero.world_model.latent_state = pred_slots.detach().clone()

    state_action_history = []
    saved = 0
    panels_by_step: Dict[int, np.ndarray] = {}
    num_slots = int(slots_np.shape[1])
    target_slot_size = int(args.obs_size)
    target_strip_w = target_slot_size * num_slots
    resize_panel_rasters = args.ocr_backend != "slotcontrast"
    with torch.no_grad():
        for step in range(start_step, rollout_end_exclusive):
            if args.policy_version == "sampled":
                action_np = np.asarray(actions_np[step], dtype=np.float32).reshape(-1)
                action_tensor = torch.from_numpy(action_np).to(device=device, dtype=torch.float32).unsqueeze(0)
            else:
                action_tensor = torch.tensor([int(actions_np[step])], dtype=torch.long, device=device)
            state_action_history.append((pred_slots, action_tensor))
            network_output = unizero.recurrent_inference(
                state_action_history=state_action_history,
                simulation_index=0,
                search_depth=[len(state_action_history)],
            )
            pred_slots = network_output.latent_state.detach()

            obs_idx = step + frame_offset
            obs_rgb = prep_obs(obs_np[obs_idx], obs_size=args.obs_size)

            gt_slot_idx = obs_idx if obs_idx < len(slots_np) else len(slots_np) - 1
            gt_slots = torch.from_numpy(slots_np[gt_slot_idx:gt_slot_idx + 1]).to(device=device, dtype=torch.float32)

            gt_strip = render_slot_strip(
                ocr_model, obs_rgb, gt_slots, device, args.ocr_backend, args.dinosaur_hard_masks
            )
            dyn_strip = render_slot_strip(
                ocr_model, obs_rgb, pred_slots, device, args.ocr_backend, args.dinosaur_hard_masks
            )

            panel = build_step_panel(
                step=step,
                gt_strip=gt_strip,
                dyn_strip=dyn_strip,
                obs_rgb=obs_rgb,
                tile_h=target_slot_size,
                strip_w=target_strip_w,
                num_slots=num_slots,
                highlight_slots=args.highlight_slots,
                highlight_color_map=args.highlight_color_map,
                resize_rasters=resize_panel_rasters,
            )
            panels_by_step[step] = panel

            out_path = Path(args.output_dir) / args.step_filename_template.format(step=step)
            ok = cv2.imwrite(str(out_path), panel)
            if not ok:
                raise RuntimeError(f"Failed to write step panel: {out_path}")
            saved += 1

    if args.merge_steps is not None and len(args.merge_steps) > 0:
        merged_out = Path(args.output_dir) / args.merged_output_name
        save_merged_selected_steps(panels_by_step, args.merge_steps, merged_out)
        print(f"Saved merged selected steps: {merged_out}")

    print(
        f"Saved {saved} step panels to {args.output_dir}. "
        f"start_step={start_step}, rollout_end_exclusive={rollout_end_exclusive}, frame_offset={frame_offset}."
    )


if __name__ == "__main__":
    main()

