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
from zoo.ocr.slate.slate import SLATE
from zoo.ocr.tools import obs_to_tensor

ENV_DEFAULTS = {
    "ocrl": {
        "ocr_config_path": "zoo/ocr/slate/config/slate_ocrl.yaml",
        "ocr_checkpoint_path": "zoo/ocr/slate_weights/slate_ocrl.pth",
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
    },
    "causal_world": {
        "ocr_config_path": "zoo/ocr/slate/config/slate_3d.yaml",
        "ocr_checkpoint_path": "zoo/ocr/slate_weights/slate_3d.pth",
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
    },
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Visualize GT vs dynamics slots for UniZero rollout."
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
        choices=["ocrl", "causal_world"],
        help="Environment preset to select model/OCR defaults.",
    )
    parser.add_argument("--model-path", type=str, default="oc_agents_weights/oz_stica_inter_seed7.pth.tar")
    parser.add_argument("--sa-slots-path", type=str, default="visuals/oz_policy_log/sa_slots.npy")
    parser.add_argument("--actions-path", type=str, default="visuals/oz_policy_log/actions.npy")
    parser.add_argument("--frames-dir", type=str, default="visuals/video_frames")
    parser.add_argument("--frame-glob", type=str, default="frame_*.jpg")
    parser.add_argument("--frame-size", type=int, default=64)
    parser.add_argument(
        "--render-slot-size",
        type=int,
        default=128,
        help="Output slot size in final panel images (render-only).",
    )
    parser.add_argument(
        "--frame-index-offset",
        type=int,
        default=-1,
        help="obs index for step t is t+offset. -1 means auto.",
    )
    parser.add_argument("--start-step", type=int, default=0)
    parser.add_argument(
        "--panel-slot-size",
        type=int,
        default=128,
        help="Rendered size (H=W) for each slot tile in output panels.",
    )
    parser.add_argument("--output-dir", type=str, default="visuals/unizero_pred_rollout_samples")
    parser.add_argument("--step-filename-template", type=str, default="step_{step:04d}.jpg")
    parser.add_argument(
        "--merge-steps",
        type=int,
        nargs="+",
        default=[0, 4, 8],
        help="Optional list of step indices to merge into one image.",
    )
    parser.add_argument("--merged-output-name", type=str, default="selected_steps_overview.jpg")
    parser.add_argument("--ocr-config-path", type=str, default="")
    parser.add_argument("--ocr-checkpoint-path", type=str, default="")
    parser.add_argument("--device", type=str, default="cuda")
    args = parser.parse_args()
    env_defaults = ENV_DEFAULTS[args.env_type]
    if not args.ocr_config_path:
        args.ocr_config_path = env_defaults["ocr_config_path"]
    if not args.ocr_checkpoint_path:
        args.ocr_checkpoint_path = env_defaults["ocr_checkpoint_path"]
    return args


def load_slate(ocr_config_path: str, checkpoint_path: str, obs_size: int, obs_channels: int, device: str) -> SLATE:
    ocr_config = OmegaConf.load(ocr_config_path)
    env_config = namedtuple("EnvConfig", ["obs_size", "obs_channels"])(obs_size=obs_size, obs_channels=obs_channels)
    slate = SLATE(ocr_config, env_config, observation_space=None, preserve_slot_order=True)
    state_dict = torch.load(checkpoint_path, map_location=device)["ocr_module_state_dict"]
    slate._module.load_state_dict(state_dict)
    slate.to(device)
    slate.eval()
    slate.requires_grad_(False)
    return slate


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
        raise RuntimeError(f"Unexpected checkpoint format at {model_path}")

    normalized = {}
    for k, v in state_dict.items():
        normalized[k[len("_orig_mod.") :]] = v if k.startswith("_orig_mod.") else v
    model.load_state_dict(normalized, strict=True)
    return model


def load_frames(frames_dir: str, frame_glob: str, frame_size: int) -> List[np.ndarray]:
    frame_paths = sorted(Path(frames_dir).glob(frame_glob))
    if len(frame_paths) < 2:
        raise FileNotFoundError(f"Need at least 2 frames in {frames_dir} matching {frame_glob}")
    frames_rgb = []
    # Skip duplicated frame_0000.
    for p in frame_paths[1:]:
        bgr = cv2.imread(str(p), cv2.IMREAD_COLOR)
        if bgr is None:
            raise RuntimeError(f"Failed to read frame: {p}")
        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        if rgb.shape[:2] != (frame_size, frame_size):
            rgb = cv2.resize(rgb, (frame_size, frame_size), interpolation=cv2.INTER_AREA)
        frames_rgb.append(rgb)
    return frames_rgb


def infer_frame_offset(num_actions: int, num_frames: int, arg_offset: int) -> int:
    if arg_offset >= 0:
        return arg_offset
    if num_frames == num_actions + 1:
        return 1
    return 0


def render_slot_strip(slate: SLATE, obs_rgb: np.ndarray, slots_1x: torch.Tensor, device: str) -> np.ndarray:
    obs = obs_to_tensor(obs_rgb[np.newaxis, ...], device=device)
    sample_rgb = slate._module.get_samples(obs, prev_slots=slots_1x)["samples"][0]
    return sample_rgb


def build_step_panel(
    step: int, gt_strip: np.ndarray, dyn_strip: np.ndarray, obs_rgb: np.ndarray, render_slot_size: int
) -> np.ndarray:
    # Upscale only visualization strips; never downscale here.
    if render_slot_size > gt_strip.shape[0]:
        scale = render_slot_size / float(gt_strip.shape[0])
        new_w = max(1, int(round(gt_strip.shape[1] * scale)))
        gt_strip = cv2.resize(gt_strip, (new_w, render_slot_size), interpolation=cv2.INTER_NEAREST)
        dyn_strip = cv2.resize(dyn_strip, (new_w, render_slot_size), interpolation=cv2.INTER_NEAREST)

    row_gap = 8
    side_pad = 6
    left_label_w = 32
    step_text = f"Step {step}"
    step_font_scale = 0.80
    step_thickness = 1
    (_, text_h), text_baseline = cv2.getTextSize(
        step_text, cv2.FONT_HERSHEY_SIMPLEX, step_font_scale, step_thickness
    )
    # Keep extra headroom so Step label never overlaps image rows.
    title_h = text_h + text_baseline + 8
    h = gt_strip.shape[0]
    w = gt_strip.shape[1]
    obs_tile_rgb = cv2.resize(obs_rgb, (h, h), interpolation=cv2.INTER_AREA)
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

    # Left labels as vertical words.
    gt_y_center = y0 + h // 2
    dyn_y_center = y1 + h // 2

    def _put_rotated_word(word: str, y_center: int) -> None:
        font_scale = 0.80
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
    device = args.device if torch.cuda.is_available() and args.device.startswith("cuda") else "cpu"
    os.makedirs(args.output_dir, exist_ok=True)

    sa_slots = np.load(args.sa_slots_path)
    actions = np.load(args.actions_path)
    frames_rgb = load_frames(args.frames_dir, args.frame_glob, args.frame_size)

    if sa_slots.ndim != 3:
        raise ValueError(f"Expected sa slots shape (T, num_slots, slot_dim), got {sa_slots.shape}")
    if args.policy_version == "sampled":
        if actions.ndim != 2:
            raise ValueError(
                f"Expected sampled continuous actions shape (T, action_dim), got {actions.shape}."
            )
    else:
        if actions.ndim != 1 or not np.issubdtype(actions.dtype, np.integer):
            raise ValueError(
                f"Expected discrete actions shape (T,), got {actions.shape} dtype={actions.dtype}"
            )

    start_step = int(args.start_step)
    if start_step < 0 or start_step >= len(sa_slots):
        raise ValueError(f"start-step={start_step} out of range for sa_slots length={len(sa_slots)}")

    unizero = load_unizero_model(
        args.model_path,
        device=device,
        policy_version=args.policy_version,
        env_type=args.env_type,
    )
    slate = load_slate(args.ocr_config_path, args.ocr_checkpoint_path, args.frame_size, 3, device)

    frame_offset = infer_frame_offset(len(actions), len(frames_rgb), args.frame_index_offset)
    rollout_end_exclusive = min(len(actions), len(frames_rgb) - frame_offset)
    if start_step >= rollout_end_exclusive:
        raise ValueError(
            f"start-step={start_step} leaves no rollout. "
            f"rollout_end_exclusive={rollout_end_exclusive}, frame_offset={frame_offset}"
        )

    pred_slots = torch.from_numpy(sa_slots[start_step:start_step + 1]).to(device=device, dtype=torch.float32)
    if not hasattr(unizero.world_model, "latent_state"):
        unizero.world_model.latent_state = pred_slots.detach().clone()

    history = []
    saved = 0
    panels_by_step: Dict[int, np.ndarray] = {}
    num_slots = int(sa_slots.shape[1])
    target_slot_size = int(args.panel_slot_size)
    if target_slot_size <= 0:
        raise ValueError(f"--panel-slot-size must be positive, got {target_slot_size}")
    target_strip_w = target_slot_size * num_slots
    with torch.no_grad():
        for step in range(start_step, rollout_end_exclusive):
            if args.policy_version == "sampled":
                action_np = np.asarray(actions[step], dtype=np.float32).reshape(-1)
                action_tensor = torch.from_numpy(action_np).to(device=device, dtype=torch.float32).unsqueeze(0)
            else:
                action_tensor = torch.tensor([int(actions[step])], dtype=torch.long, device=device)
            history.append((pred_slots, action_tensor))
            output = unizero.recurrent_inference(
                state_action_history=history,
                simulation_index=0,
                search_depth=[len(history)],
            )
            pred_slots = output.latent_state.detach()

            obs_idx = step + frame_offset
            obs_rgb = frames_rgb[obs_idx]

            gt_slot_idx = obs_idx if obs_idx < len(sa_slots) else len(sa_slots) - 1
            gt_slots = torch.from_numpy(sa_slots[gt_slot_idx:gt_slot_idx + 1]).to(device=device, dtype=torch.float32)

            gt_strip = render_slot_strip(slate, obs_rgb, gt_slots, device)
            dyn_strip = render_slot_strip(slate, obs_rgb, pred_slots, device)
            if gt_strip.shape[0] != target_slot_size or gt_strip.shape[1] != target_strip_w:
                gt_strip = cv2.resize(gt_strip, (target_strip_w, target_slot_size), interpolation=cv2.INTER_NEAREST)
            if dyn_strip.shape[0] != target_slot_size or dyn_strip.shape[1] != target_strip_w:
                dyn_strip = cv2.resize(dyn_strip, (target_strip_w, target_slot_size), interpolation=cv2.INTER_NEAREST)
            panel = build_step_panel(
                step=step,
                gt_strip=gt_strip,
                dyn_strip=dyn_strip,
                obs_rgb=obs_rgb,
                render_slot_size=args.render_slot_size,
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
