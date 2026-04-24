import argparse
import os
from collections import namedtuple
from pathlib import Path
from typing import List

import cv2
import numpy as np
import torch
from easydict import EasyDict
from omegaconf import OmegaConf

from lzero.model.unizero_model import UniZeroModel
from zoo.ocr.slate.slate import SLATE
from zoo.ocr.tools import obs_to_tensor


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Visualize UniZero predicted slot rollouts with SLATE get_samples (OCRL discrete)."
    )
    parser.add_argument(
        "--model-path",
        type=str,
        default="oc_agents_weights/stica_obj_goal_seed=0.pth.tar",
        help="Path to trained UniZero policy checkpoint.",
    )
    parser.add_argument(
        "--sa-slots-path",
        type=str,
        default="visuals/oz_policy_log/sa_slots.npy",
        help="Path to saved GT slots from eval (sa_slots_*.npy).",
    )
    parser.add_argument(
        "--actions-path",
        type=str,
        default="visuals/oz_policy_log/actions.npy",
        help="Path to saved actions from eval (actions_*.npy).",
    )
    parser.add_argument(
        "--frames-dir",
        type=str,
        default="visuals/video_frames",
        help="Directory with extracted GT frames (e.g. frame_0000.jpg...).",
    )
    parser.add_argument(
        "--frame-glob",
        type=str,
        default="frame_*.jpg",
        help="Frame filename glob in frames-dir.",
    )
    parser.add_argument(
        "--frame-size",
        type=int,
        default=64,
        help="Resize all GT frames to square size (default: 64).",
    )
    parser.add_argument(
        "--frame-index-offset",
        type=int,
        default=-1,
        help=(
            "Mapping from predicted next state at action[t] to frame index: "
            "target_idx=t+offset. Use -1 to auto-infer (prefer 1 if frames=T+1, else 0)."
        ),
    )
    parser.add_argument(
        "--start-step",
        type=int,
        default=0,
        help="Start step index in sa_slots/actions from which rollout begins.",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="visuals/unizero_pred_rollout_samples",
        help="Where visualization images are saved.",
    )
    parser.add_argument(
        "--ocr-config-path",
        type=str,
        default="zoo/ocr/slate/config/slate_ocrl.yaml",
        help="SLATE OCR config path.",
    )
    parser.add_argument(
        "--ocr-checkpoint-path",
        type=str,
        default="zoo/ocr/slate_weights/slate_ocrl.pth",
        help="SLATE OCR checkpoint path.",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda",
        help="Torch device, e.g. cuda or cpu.",
    )
    return parser.parse_args()


def load_slate(
    ocr_config_path: str,
    checkpoint_path: str,
    obs_size: int,
    obs_channels: int,
    device: str,
) -> SLATE:
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


def load_unizero_model(model_path: str, device: str) -> UniZeroModel:
    # Keep this local to avoid importing eval entrypoints (they trigger unrelated
    # env/plugin initialization noise, e.g. gym/mujoco warnings).
    action_space_size = 4
    num_slots = 6
    slot_dim = 192
    num_unroll_steps = 10
    infer_context_length = 4
    num_layers = 2
    num_heads = 8
    tokens_per_block = num_slots * 2

    world_model_cfg = EasyDict(
        dict(
            model_type="slot",
            encoder_type="resnet",
            continuous_action_space=False,
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
            num_layers=num_layers,
            num_heads=num_heads,
            embed_dim=slot_dim,
            embed_pdrop=0.1,
            resid_pdrop=0.1,
            attn_pdrop=0.1,
            support_size=601,
            max_cache_size=5000,
            env_num=1,
            latent_recon_loss_weight=0.0,
            perceptual_loss_weight=0.0,
            policy_entropy_weight=5e-3,
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
        )
    )

    model_cfg = dict(
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

    model = UniZeroModel(**model_cfg)
    model.to(device)
    model.eval()

    checkpoint = torch.load(model_path, map_location=device)
    state_dict = checkpoint["model"] if isinstance(checkpoint, dict) and "model" in checkpoint else checkpoint
    if not isinstance(state_dict, dict):
        raise RuntimeError(f"Unexpected checkpoint format at {model_path}: state_dict is not a dict.")

    # Checkpoints saved from torch.compile() may prepend "_orig_mod." to every key.
    # Strip this prefix so weights can be loaded into a regular nn.Module.
    normalized_state_dict = {}
    for key, value in state_dict.items():
        if key.startswith("_orig_mod."):
            normalized_state_dict[key[len("_orig_mod."):]] = value
        else:
            normalized_state_dict[key] = value

    state_dict = normalized_state_dict
    model.load_state_dict(state_dict, strict=True)
    return model


def load_frames(frames_dir: str, frame_glob: str, frame_size: int) -> List[np.ndarray]:
    frame_paths = sorted(Path(frames_dir).glob(frame_glob))
    if len(frame_paths) == 0:
        raise FileNotFoundError(f"No frames found in {frames_dir} with pattern {frame_glob}")
    if len(frame_paths) < 2:
        raise ValueError("Need at least 2 frames to skip duplicated first frame.")

    frames_rgb = []
    # Video export in this pipeline duplicates the very first frame.
    # We keep files untouched on disk and simply skip reading frame_0000.
    for p in frame_paths[1:]:
        bgr = cv2.imread(str(p), cv2.IMREAD_COLOR)
        if bgr is None:
            raise RuntimeError(f"Failed to read frame: {p}")
        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        if rgb.shape[0] != frame_size or rgb.shape[1] != frame_size:
            rgb = cv2.resize(rgb, (frame_size, frame_size), interpolation=cv2.INTER_AREA)
        frames_rgb.append(rgb)

    return frames_rgb


def infer_frame_offset(num_actions: int, num_frames: int, arg_offset: int) -> int:
    if arg_offset >= 0:
        return arg_offset
    if num_frames == num_actions + 1:
        return 1
    if num_frames == num_actions:
        return 0

    return 0


def main() -> None:
    args = parse_args()
    device = args.device if torch.cuda.is_available() and args.device.startswith("cuda") else "cpu"
    os.makedirs(args.output_dir, exist_ok=True)

    sa_slots = np.load(args.sa_slots_path)
    actions = np.load(args.actions_path)
    frames_rgb = load_frames(
        frames_dir=args.frames_dir,
        frame_glob=args.frame_glob,
        frame_size=args.frame_size,
    )

    if sa_slots.ndim != 3:
        raise ValueError(f"Expected sa slots of shape (T, num_slots, slot_dim), got {sa_slots.shape}")
    if actions.ndim != 1:
        raise ValueError(f"Expected 1D discrete actions, got {actions.shape}")

    if not np.issubdtype(actions.dtype, np.integer):
        raise ValueError("This script currently supports discrete OCRL only (actions must be integer type).")

    start_step = int(args.start_step)
    if start_step < 0 or start_step >= len(sa_slots):
        raise ValueError(f"start-step={start_step} out of range for sa_slots length={len(sa_slots)}")

    unizero = load_unizero_model(args.model_path, device=device)
    slate = load_slate(
        ocr_config_path=args.ocr_config_path,
        checkpoint_path=args.ocr_checkpoint_path,
        obs_size=64,
        obs_channels=3,
        device=device,
    )

    frame_offset = infer_frame_offset(num_actions=len(actions), num_frames=len(frames_rgb), arg_offset=args.frame_index_offset)
    rollout_steps = min(len(actions), len(sa_slots) - 1, len(frames_rgb) - frame_offset)
    if start_step >= rollout_steps:
        raise ValueError(
            f"start-step={start_step} leaves no rollout. "
            f"Computed rollout_steps={rollout_steps}, frame_offset={frame_offset}."
        )

    pred_slots = torch.from_numpy(sa_slots[start_step:start_step + 1]).to(device=device, dtype=torch.float32)
    # The recurrent path in WorldModel deletes `self.latent_state` before writing the new one.
    # In standalone rollout scripts this attribute may not exist yet (unlike full policy flow),
    # so initialize it once to avoid AttributeError on the first recurrent step.
    if not hasattr(unizero.world_model, "latent_state"):
        unizero.world_model.latent_state = pred_slots.detach().clone()

    state_action_history = []
    saved = 0
    with torch.no_grad():
        for step in range(start_step, rollout_steps):
            action_tensor = torch.tensor([int(actions[step])], dtype=torch.long, device=device)
            state_action_history.append((pred_slots, action_tensor))
            network_output = unizero.recurrent_inference(
                state_action_history=state_action_history,
                simulation_index=0,
                search_depth=[len(state_action_history)],
            )
            pred_slots = network_output.latent_state.detach()

            target_frame_idx = step + frame_offset
            gt_obs = obs_to_tensor(frames_rgb[target_frame_idx][np.newaxis, ...], device=device)
            sample_dict = slate._module.get_samples(gt_obs, prev_slots=pred_slots)
            sample_rgb = sample_dict["samples"][0]

            out_path = os.path.join(args.output_dir, f"pred_rollout_step_{step:04d}.jpg")
            ok = cv2.imwrite(out_path, cv2.cvtColor(sample_rgb, cv2.COLOR_RGB2BGR))
            if not ok:
                raise RuntimeError(f"Failed to write output image: {out_path}")
            saved += 1

    print(
        f"Saved {saved} rollout samples to {args.output_dir}. "
        f"start_step={start_step}, rollout_steps={rollout_steps}, frame_offset={frame_offset}."
    )


if __name__ == "__main__":
    main()
