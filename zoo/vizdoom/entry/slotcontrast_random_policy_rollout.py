import argparse
import os
from pathlib import Path

import numpy as np
import torch
from easydict import EasyDict

from zoo.ocr.slotcontrast import load_from_checkpoint as load_slotcontrast_from_checkpoint
from zoo.vizdoom.env.vizdoom_wrappers import wrap_lightzero


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run a random-policy Vizdoom rollout and save obs/slots/actions with SlotContrast."
    )
    parser.add_argument("--env-id", type=str, default="VizdoomDefendLine-v0")
    parser.add_argument("--seed", type=int, default=777)
    parser.add_argument("--max-steps", type=int, default=40)
    parser.add_argument("--ocr-config-path", type=str, default="zoo/ocr/slotcontrast/configs/vizdoom_sc.yaml")
    parser.add_argument("--checkpoint-path", type=str, default="zoo/ocr/slotcontrast_weights/slotcontrast_vizdoom.ckpt")
    parser.add_argument("--output-dir", type=str, default="visuals/random_policy_log")
    parser.add_argument("--device", type=str, default="cuda")
    return parser.parse_args()


def _build_env_config(args: argparse.Namespace) -> EasyDict:
    return EasyDict(
        dict(
            env_id=args.env_id,
            seed=args.seed,
            from_pixels=True,
            observation_shape=(3, 336, 336),
            gray_scale=False,
            transform2string=False,
            warp_frame=True,
            scale=False,
            game_wrapper=True,
            oc_model=False,
            max_episode_steps=args.max_steps,
        )
    )


def _obs_to_tensor_safe(obs: np.ndarray, device: str) -> torch.Tensor:
    obs = np.ascontiguousarray(obs)
    return torch.as_tensor(obs[np.newaxis].transpose(0, 3, 1, 2), dtype=torch.float32, device=device) / 255.0


def main() -> None:
    args = parse_args()
    device = args.device if torch.cuda.is_available() and args.device.startswith("cuda") else "cpu"
    os.makedirs(args.output_dir, exist_ok=True)

    env = wrap_lightzero(_build_env_config(args))
    env.action_space.seed(args.seed)

    slotcontrast = load_slotcontrast_from_checkpoint(
        config_path=args.ocr_config_path,
        checkpoint_path=args.checkpoint_path,
        device=device,
    )
    slotcontrast.requires_grad_(False)
    slotcontrast.eval()

    obs = env.reset()
    if isinstance(obs, tuple):
        obs = obs[0]

    obs_buffer = [np.asarray(obs, dtype=np.uint8).copy()]
    obs_t = _obs_to_tensor_safe(obs, device=device)
    slots_t = slotcontrast.extract_slots(obs_t, prev_slots=None)
    slots_buffer = [slots_t.squeeze(0).detach().cpu().numpy().astype(np.float32)]
    actions_buffer = []
    prev_slots = slots_t

    done = False
    step_count = 0
    while not done and step_count < args.max_steps:
        action = int(env.action_space.sample())
        actions_buffer.append(action)
        obs, _, done, _ = env.step(action)
        if isinstance(obs, tuple):
            obs = obs[0]
        obs_buffer.append(np.asarray(obs, dtype=np.uint8).copy())

        obs_t = _obs_to_tensor_safe(obs, device=device)
        slots_t = slotcontrast.extract_slots(obs_t, prev_slots=prev_slots)
        slots_buffer.append(slots_t.squeeze(0).detach().cpu().numpy().astype(np.float32))
        prev_slots = slots_t
        step_count += 1

    env.close()

    obs_np = np.stack(obs_buffer, axis=0)
    slots_np = np.stack(slots_buffer, axis=0)
    actions_np = np.asarray(actions_buffer, dtype=np.int64)

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
