import argparse
import os
from pathlib import Path

import numpy as np
if not hasattr(np, 'bool'):
    np.bool = bool
if not hasattr(np, 'int'):
    np.int = int
if not hasattr(np, 'float'):
    np.float = float
import torch
from gym.wrappers import TimeLimit
from omegaconf import OmegaConf

from zoo.causal_world.env.causal_world.cw_envs import CwTargetEnv
from zoo.ocr.slate.slate import SLATE
from zoo.ocr.tools import obs_to_tensor


def _obs_to_tensor_safe(obs: np.ndarray, device: str) -> torch.Tensor:
    obs = np.ascontiguousarray(obs)
    return obs_to_tensor(obs[np.newaxis], device=device)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run a random-policy CausalWorld rollout and save obs/slots/actions."
    )
    parser.add_argument(
        "--env-config-path",
        type=str,
        default="zoo/causal_world/env/causal_world/cw_envs/config/reaching-hard_orig.yaml",
    )
    parser.add_argument("--seed", type=int, default=142)
    parser.add_argument("--max-steps", type=int, default=125)
    parser.add_argument("--ocr-config-path", type=str, default="zoo/ocr/slate/config/slate_3d.yaml")
    parser.add_argument("--checkpoint-path", type=str, default="zoo/ocr/slate_weights/slate_3d.pth")
    parser.add_argument("--output-dir", type=str, default="visuals/random_policy_log")
    parser.add_argument("--device", type=str, default="cuda")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    device = args.device if torch.cuda.is_available() and args.device.startswith("cuda") else "cpu"
    os.makedirs(args.output_dir, exist_ok=True)

    env_config = OmegaConf.load(args.env_config_path)
    env = CwTargetEnv(env_config, args.seed)
    env.action_space.seed(args.seed)
    env = TimeLimit(env, env.unwrapped._max_episode_length)

    config_ocr = OmegaConf.load(args.ocr_config_path)
    slate = SLATE(config_ocr, env_config, observation_space=None, preserve_slot_order=True)
    slate.to(device)
    state_dict = torch.load(args.checkpoint_path, map_location=device)["ocr_module_state_dict"]
    slate._module.load_state_dict(state_dict)
    slate.requires_grad_(False)
    slate.eval()

    obs = env.reset()
    if isinstance(obs, tuple):
        obs = obs[0]

    obs_buffer = [np.asarray(obs, dtype=np.uint8).copy()]
    obs_t = _obs_to_tensor_safe(obs, device=device)
    slots_t = slate._module._get_slots(obs_t, prev_slots=None)
    slots_buffer = [slots_t.squeeze(0).detach().cpu().numpy().astype(np.float32)]
    actions_buffer = []
    prev_slots = slots_t

    done = False
    step_count = 0
    while not done and step_count < args.max_steps:
        action = np.asarray(env.action_space.sample(), dtype=np.float32)
        actions_buffer.append(action.copy())
        obs, _, done, _ = env.step(action)
        if isinstance(obs, tuple):
            obs = obs[0]
        obs_buffer.append(np.asarray(obs, dtype=np.uint8).copy())

        obs_t = _obs_to_tensor_safe(obs, device=device)
        slots_t = slate._module._get_slots(obs_t, prev_slots=prev_slots)
        slots_buffer.append(slots_t.squeeze(0).detach().cpu().numpy().astype(np.float32))
        prev_slots = slots_t
        step_count += 1

    env.close()

    obs_np = np.stack(obs_buffer, axis=0)      # (T, H, W, C)
    slots_np = np.stack(slots_buffer, axis=0)  # (T, num_slots, slot_dim)
    actions_np = np.stack(actions_buffer, axis=0).astype(np.float32) if actions_buffer else np.zeros(
        (0, int(env.action_space.shape[0])), dtype=np.float32
    )

    obs_path = Path(args.output_dir) / "random_obs.npy"
    slots_path = Path(args.output_dir) / "random_slots.npy"
    actions_path = Path(args.output_dir) / "random_actions.npy"
    np.save(obs_path, obs_np)
    np.save(slots_path, slots_np)
    np.save(actions_path, actions_np)

    print("Saved random policy rollout:")
    print(f"  obs:     {obs_path} shape={obs_np.shape}")
    print(f"  slots:   {slots_path} shape={slots_np.shape}")
    print(f"  actions: {actions_path} shape={actions_np.shape}")


if __name__ == "__main__":
    main()
