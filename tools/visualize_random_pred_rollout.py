import argparse
import os
from collections import namedtuple

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
        description="Visualize UniZero predicted rollout using random-policy obs/slots/actions dumps."
    )
    parser.add_argument("--model-path", type=str, default="oc_agents_weights/stica_obj_goal_seed=0.pth.tar")
    parser.add_argument(
        "--obs-path",
        type=str,
        default="visuals/random_policy_log/random_obs.npy",
        help="Path to random-policy obs dump, shape (T, H, W, C).",
    )
    parser.add_argument(
        "--slots-path",
        type=str,
        default="visuals/random_policy_log/random_slots.npy",
        help="Path to random-policy slots dump, shape (T, num_slots, slot_dim).",
    )
    parser.add_argument(
        "--actions-path",
        type=str,
        default="visuals/random_policy_log/random_actions.npy",
        help="Path to random-policy actions dump, shape (T-1,).",
    )
    parser.add_argument("--start-step", type=int, default=0)
    parser.add_argument(
        "--frame-index-offset",
        type=int,
        default=-1,
        help="Mapping from predicted state at action[t] to obs index t+offset. -1 means auto.",
    )
    parser.add_argument("--obs-size", type=int, default=64, help="Resize GT obs to square size.")
    parser.add_argument("--output-dir", type=str, default="visuals/random_policy_pred_rollout_samples")
    parser.add_argument("--output-prefix", type=str, default="random_pred_rollout")
    parser.add_argument("--ocr-config-path", type=str, default="zoo/ocr/slate/config/slate_ocrl.yaml")
    parser.add_argument("--ocr-checkpoint-path", type=str, default="zoo/ocr/slate_weights/slate_ocrl.pth")
    parser.add_argument("--device", type=str, default="cuda")
    return parser.parse_args()


def load_slate(
    ocr_config_path: str,
    checkpoint_path: str,
    obs_size: int,
    obs_channels: int,
    device: str,
) -> SLATE:
    ocr_config = OmegaConf.load(ocr_config_path)
    env_config = namedtuple("EnvConfig", ["obs_size", "obs_channels"])(obs_size=obs_size, obs_channels=obs_channels)

    slate = SLATE(ocr_config, env_config, observation_space=None, preserve_slot_order=True)
    state_dict = torch.load(checkpoint_path, map_location=device)["ocr_module_state_dict"]
    slate._module.load_state_dict(state_dict)
    slate.to(device)
    slate.eval()
    slate.requires_grad_(False)
    return slate


def load_unizero_model(model_path: str, device: str) -> UniZeroModel:
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

    normalized_state_dict = {}
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
        obs = cv2.resize(obs, (obs_size, obs_size), interpolation=cv2.INTER_AREA)
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


def main() -> None:
    args = parse_args()
    device = args.device if torch.cuda.is_available() and args.device.startswith("cuda") else "cpu"
    os.makedirs(args.output_dir, exist_ok=True)

    obs_np = np.load(args.obs_path)
    slots_np = np.load(args.slots_path)
    actions_np = np.load(args.actions_path)

    if obs_np.ndim != 4:
        raise ValueError(f"Expected obs shape (T, H, W, C), got {obs_np.shape}")
    if slots_np.ndim != 3:
        raise ValueError(f"Expected slots shape (T, num_slots, slot_dim), got {slots_np.shape}")
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

    unizero = load_unizero_model(args.model_path, device=device)
    slate = load_slate(
        ocr_config_path=args.ocr_config_path,
        checkpoint_path=args.ocr_checkpoint_path,
        obs_size=args.obs_size,
        obs_channels=3,
        device=device,
    )

    pred_slots = torch.from_numpy(slots_np[start_step:start_step + 1]).to(device=device, dtype=torch.float32)
    if not hasattr(unizero.world_model, "latent_state"):
        unizero.world_model.latent_state = pred_slots.detach().clone()

    state_action_history = []
    saved = 0
    with torch.no_grad():
        for step in range(start_step, rollout_end_exclusive):
            action_tensor = torch.tensor([int(actions_np[step])], dtype=torch.long, device=device)
            state_action_history.append((pred_slots, action_tensor))
            network_output = unizero.recurrent_inference(
                state_action_history=state_action_history,
                simulation_index=0,
                search_depth=[len(state_action_history)],
            )
            pred_slots = network_output.latent_state.detach()

            target_obs_idx = step + frame_offset
            gt_obs_hwc = prep_obs(obs_np[target_obs_idx], obs_size=args.obs_size)
            gt_obs_t = obs_to_tensor(gt_obs_hwc[np.newaxis, ...], device=device)
            sample_rgb = slate._module.get_samples(gt_obs_t, prev_slots=pred_slots)["samples"][0]

            out_path = os.path.join(args.output_dir, f"{args.output_prefix}_step_{step:04d}.jpg")
            ok = cv2.imwrite(out_path, cv2.cvtColor(sample_rgb, cv2.COLOR_RGB2BGR))
            if not ok:
                raise RuntimeError(f"Failed to write output image: {out_path}")
            saved += 1

    print(
        f"Saved {saved} random-policy rollout visualizations to {args.output_dir}. "
        f"start_step={start_step}, frame_offset={frame_offset}, rollout_end_exclusive={rollout_end_exclusive}."
    )


if __name__ == "__main__":
    main()

