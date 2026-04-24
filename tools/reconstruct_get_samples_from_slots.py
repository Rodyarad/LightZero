import argparse
import os
from collections import namedtuple
from pathlib import Path

import cv2
import numpy as np
import torch
from omegaconf import OmegaConf

from zoo.ocr.slate.slate import SLATE
from zoo.ocr.tools import obs_to_tensor


def load_slate(
    ocr_config_path: str,
    checkpoint_path: str,
    obs_size: int = 64,
    obs_channels: int = 3,
    device: str = "cuda",
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


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Visualize slots via SLATE get_samples using GT images."
    )
    parser.add_argument(
        "--slots-path",
        type=str,
        default="visuals/oz_policy_log/sa_slots.npy",
        help="Path to slots .npy (T, num_slots, slot_dim).",
    )
    parser.add_argument(
        "--frames-dir",
        type=str,
        default="visuals/video_frames",
        help="Directory with extracted GT frames.",
    )
    parser.add_argument("--frame-glob", type=str, default="frame_*.jpg", help="Glob pattern for frames in frames-dir.")
    parser.add_argument(
        "--output-dir",
        type=str,
        default="visuals/sa_slots_get_samples",
        help="Directory for output images.",
    )
    parser.add_argument("--slot-prefix", type=str, default="slots", help="Prefix for output filenames.")
    parser.add_argument("--obs-size", type=int, default=64, help="Resize GT frames to this square size.")
    parser.add_argument(
        "--frame-offset",
        type=int,
        default=-1,
        help="Frame index offset for slot t. -1 means auto: use 1 if frames=T+1 else 0.",
    )
    parser.add_argument(
        "--keep-first-frame",
        action="store_true",
        help="Do not skip frame_0000. By default the first frame is skipped.",
    )
    parser.add_argument("--ocr-config-path", type=str, default="zoo/ocr/slate/config/slate_ocrl.yaml")
    parser.add_argument("--ocr-checkpoint-path", type=str, default="zoo/ocr/slate_weights/slate_ocrl.pth")
    parser.add_argument("--device", type=str, default="cuda")
    return parser.parse_args()


def load_frames(frames_dir: str, frame_glob: str, obs_size: int, skip_first_frame: bool) -> list:
    frame_paths = sorted(Path(frames_dir).glob(frame_glob))
    if len(frame_paths) == 0:
        raise FileNotFoundError(f"No frames found in {frames_dir} with pattern {frame_glob}")
    if skip_first_frame:
        frame_paths = frame_paths[1:]

    frames = []
    for p in frame_paths:
        img_bgr = cv2.imread(str(p), cv2.IMREAD_COLOR)
        if img_bgr is None:
            raise RuntimeError(f"Failed to read frame: {p}")
        img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
        if img_rgb.shape[0] != obs_size or img_rgb.shape[1] != obs_size:
            img_rgb = cv2.resize(img_rgb, (obs_size, obs_size), interpolation=cv2.INTER_AREA)
        frames.append(img_rgb)
    return frames


def resolve_frame_offset(num_slots_steps: int, num_frames: int, frame_offset: int) -> int:
    if frame_offset >= 0:
        return frame_offset
    if num_frames == num_slots_steps + 1:
        return 1
    return 0


def main() -> None:
    args = parse_args()
    device = args.device if torch.cuda.is_available() and args.device.startswith("cuda") else "cpu"
    os.makedirs(args.output_dir, exist_ok=True)

    slots = np.load(args.slots_path)
    if slots.ndim != 3:
        raise ValueError(f"Expected slots shape (T, num_slots, slot_dim), got {slots.shape}")

    frames = load_frames(
        frames_dir=args.frames_dir,
        frame_glob=args.frame_glob,
        obs_size=args.obs_size,
        skip_first_frame=not args.keep_first_frame,
    )
    offset = resolve_frame_offset(num_slots_steps=len(slots), num_frames=len(frames), frame_offset=args.frame_offset)
    usable_steps = min(len(slots), len(frames) - offset)
    if usable_steps <= 0:
        raise ValueError(
            f"No aligned data: slots={len(slots)}, frames={len(frames)}, frame_offset={offset}."
        )

    slate = load_slate(
        ocr_config_path=args.ocr_config_path,
        checkpoint_path=args.ocr_checkpoint_path,
        obs_size=args.obs_size,
        obs_channels=3,
        device=device,
    )

    with torch.no_grad():
        for t in range(usable_steps):
            frame_rgb = frames[t + offset]
            obs = obs_to_tensor(frame_rgb[np.newaxis, ...], device=device)
            prev_slots = torch.from_numpy(slots[t:t + 1]).to(device=device, dtype=torch.float32)
            sample = slate._module.get_samples(obs, prev_slots=prev_slots)["samples"][0]

            out_path = os.path.join(args.output_dir, f"{args.slot_prefix}_sample_{t:04d}.jpg")
            ok = cv2.imwrite(out_path, cv2.cvtColor(sample, cv2.COLOR_RGB2BGR))
            if not ok:
                raise RuntimeError(f"Failed to write {out_path}")

    print(
        f"Saved {usable_steps} get_samples visualizations to {args.output_dir} "
        f"(frame_offset={offset}, skip_first_frame={not args.keep_first_frame})."
    )


if __name__ == "__main__":
    main()

