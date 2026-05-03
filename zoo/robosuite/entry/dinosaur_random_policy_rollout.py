import argparse
import os
from pathlib import Path

import numpy as np
import torch
from easydict import EasyDict

from zoo.ocr.tools import Dinosaur
from zoo.robosuite.env.robosuite_wrappers import SlotExtractor, wrap_lightzero


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run a random-policy Robosuite rollout and save obs/slots/actions with DINOSAUR."
    )
    parser.add_argument("--seed", type=int, default=777)
    parser.add_argument("--max-steps", type=int, default=125)
    parser.add_argument("--checkpoint-path", type=str, default="zoo/ocr/dinosaur_weights/robosuite.ckpt")
    parser.add_argument("--output-dir", type=str, default="visuals/random_policy_log")
    parser.add_argument("--device", type=str, default="cuda")
    return parser.parse_args()


def _build_env_config(args: argparse.Namespace) -> EasyDict:
    return EasyDict(
        dict(
            seed=args.seed,
            from_pixels=True,
            observation_shape=(3, 224, 224),
            gray_scale=False,
            transform2string=False,
            warp_frame=True,
            scale=False,
            max_episode_steps=args.max_steps,
            oc_model=False,
        )
    )


def _extract_slots(slot_extractor: SlotExtractor, obs: np.ndarray, prev_slots: np.ndarray):
    if prev_slots is None:
        prev_slots = slot_extractor(obs, prev_slots=None)
    slots = slot_extractor(obs, prev_slots=prev_slots)
    return slots, slots


def main() -> None:
    args = parse_args()
    device = args.device if torch.cuda.is_available() and args.device.startswith("cuda") else "cpu"
    os.makedirs(args.output_dir, exist_ok=True)

    env = wrap_lightzero(_build_env_config(args))
    env.action_space.seed(args.seed)

    dinosaur = Dinosaur(
        dino_model_name="vit_base_patch16_224_dino",
        n_slots=5,
        slot_dim=64,
        intput_feature_dim=768,
        num_patches=196,
        features=(2048, 2048, 2048),
    )
    state_dict = torch.load(args.checkpoint_path, map_location=device)["state_dict"]
    state_dict = {key[len("models."):]: value for key, value in state_dict.items()}
    dinosaur.load_state_dict(state_dict)
    dinosaur = dinosaur.eval()
    dinosaur.requires_grad_(False)
    slot_extractor = SlotExtractor(model=dinosaur, device=device, name_model="DINOSAUR")

    obs = env.reset()
    if isinstance(obs, tuple):
        obs = obs[0]

    obs_buffer = [np.asarray(obs, dtype=np.uint8).copy()]
    slots, prev_slots = _extract_slots(slot_extractor, obs, prev_slots=None)
    slots_buffer = [slots.astype(np.float32)]
    actions_buffer = []

    done = False
    step_count = 0
    while not done and step_count < args.max_steps:
        action = np.asarray(env.action_space.sample(), dtype=np.float32)
        actions_buffer.append(action.copy())
        obs, _, done, _ = env.step(action)
        if isinstance(obs, tuple):
            obs = obs[0]
        obs_buffer.append(np.asarray(obs, dtype=np.uint8).copy())

        slots, prev_slots = _extract_slots(slot_extractor, obs, prev_slots=prev_slots)
        slots_buffer.append(slots.astype(np.float32))
        step_count += 1

    env.close()

    obs_np = np.stack(obs_buffer, axis=0)
    slots_np = np.stack(slots_buffer, axis=0)
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
