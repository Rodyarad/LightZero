"""Offline ray-traced video replay for ManiSkill trajectories.

This script reads a (.npz, .json) pair produced by ``maniskill_eval.py --save-trajectory``,
creates a fresh ManiSkill env with the requested SAPIEN shader (e.g. ``rt-fast``) and
``obs_mode='none'`` so ray tracing is allowed, then re-executes the saved actions and
writes a video of ``env.render()`` (which uses the separate human-render camera).

Workflow:

1. Run eval with trajectory dumping:
   ``python zoo/maniskill/entry/maniskill_eval.py --save-trajectory --model-path ...``
2. Replay one (or all) episodes with ray tracing:
   ``python zoo/maniskill/entry/maniskill_rt_replay.py \
       --trajectory ./visuals/trajectory/episode_0000_seed35100_*.npz \
       --shader rt-fast --out ./visuals/rt_replay.mp4``

Requirements for ``rt*`` shaders: a CUDA-capable NVIDIA GPU with OptiX support.
Falls back to ``default``+ shadows works on any GPU/CPU (will look like the eval render).
"""
from __future__ import annotations

import argparse
import glob
import json
import os
from pathlib import Path
from typing import Iterable

import cv2
import gymnasium
import numpy as np

import mani_skill.envs  # noqa: F401  -- registers built-in tasks
from mani_skill.utils import sapien_utils
from mani_skill.utils.structs.pose import Pose
from zoo.maniskill.env import maniskill3  # noqa: F401  -- registers PushCubeCustom-v1

# PushCubeEnv default sensor camera ("base_camera") — same view the LightZero agent sees during eval.
# Source: mani_skill/envs/tasks/tabletop/push_cube.py:_default_sensor_configs
_EVAL_CAMERA_EYE = [0.3, 0.0, 0.6]
_EVAL_CAMERA_TARGET = [-0.1, 0.0, 0.1]
_EVAL_CAMERA_FOV = float(np.pi / 2)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Replay a saved ManiSkill trajectory with a chosen SAPIEN shader.")
    parser.add_argument(
        "--trajectory",
        type=str,
        required=True,
        help="Path to a single .npz trajectory file, or a glob pattern matching multiple .npz files.",
    )
    parser.add_argument(
        "--out",
        type=str,
        default="./visuals/rt_replay",
        help="Output mp4 file (single trajectory) or output directory (multiple trajectories).",
    )
    parser.add_argument(
        "--shader",
        type=str,
        default="rt-fast",
        choices=["default", "rt", "rt-fast", "rt-med"],
        help="SAPIEN shader. 'rt*' gives photo-realistic ray-traced video (needs NVIDIA OptiX).",
    )
    parser.add_argument("--image-size", type=int, default=672, help="Render resolution (square).")
    parser.add_argument("--fps", type=int, default=20, help="Output video FPS.")
    parser.add_argument(
        "--sim-backend",
        type=str,
        default="cpu",
        choices=["cpu", "gpu", "auto"],
        help="Sim backend. RT shaders only work on CPU sim backend in ManiSkill 3.",
    )
    parser.add_argument(
        "--enable-shadow",
        dest="enable_shadow",
        action="store_true",
        default=True,
        help="Enable shadows under the rasterized 'default' shader. Ignored under RT.",
    )
    parser.add_argument(
        "--no-shadow",
        dest="enable_shadow",
        action="store_false",
        help="Disable shadows under the rasterized 'default' shader.",
    )
    parser.add_argument(
        "--camera",
        type=str,
        default="eval",
        choices=["eval", "default"],
        help=(
            "Which human-render camera to use for the video. 'eval' matches the sensor camera "
            "the agent saw (front view). 'default' uses ManiSkill's stock side-angle render camera."
        ),
    )
    parser.add_argument(
        "--save-frames",
        action="store_true",
        help="Save individual frames as image files instead of (or in addition to) an mp4. "
             "Pair with --frames-format to pick png/jpg.",
    )
    parser.add_argument(
        "--frames-format",
        type=str,
        default="png",
        choices=["png", "jpg"],
        help="Image format used when --save-frames is set. png = lossless (default).",
    )
    parser.add_argument(
        "--no-mp4",
        dest="save_mp4",
        action="store_false",
        default=True,
        help="Skip mp4 writing. Useful together with --save-frames if you only want frames.",
    )
    return parser.parse_args()


def _frame_to_uint8_hwc(img) -> np.ndarray:
    """Convert ManiSkill render output (torch.Tensor or numpy, possibly batched) to (H,W,3) uint8."""
    if hasattr(img, "detach"):
        img = img.detach().cpu().numpy()
    img = np.asarray(img)
    if img.ndim == 4:
        img = img[0]
    if img.shape[-1] == 4:
        img = img[..., :3]
    if img.dtype != np.uint8:
        img = np.clip(img, 0, 255).astype(np.uint8)
    return img


def _resolve_trajectories(pattern: str) -> list[str]:
    if os.path.isfile(pattern):
        return [pattern]
    matches = sorted(glob.glob(pattern))
    if not matches:
        raise FileNotFoundError(f"No trajectory file matched: {pattern!r}")
    return matches


def _load_meta(npz_path: str) -> dict:
    meta_path = npz_path[:-4] + ".json"
    if not os.path.isfile(meta_path):
        raise FileNotFoundError(f"Missing sidecar JSON for {npz_path}: expected {meta_path}")
    with open(meta_path, "r") as f:
        return json.load(f)


def replay_one(
    npz_path: str,
    out_path: str,
    shader: str,
    image_size: int,
    fps: int,
    sim_backend: str,
    enable_shadow: bool,
    camera: str = "eval",
    save_mp4: bool = True,
    save_frames: bool = False,
    frames_dir: str | None = None,
    frames_format: str = "png",
) -> None:
    data = np.load(npz_path, allow_pickle=False)
    actions = np.asarray(data["actions"], dtype=np.float32)
    seed_array = np.asarray(data["seed"])
    seed = int(seed_array.item()) if seed_array.ndim == 0 else int(seed_array[0])

    meta = _load_meta(npz_path)
    env_id = meta.get("env_id", "PushCubeCustom-v1")
    control_mode = meta.get("control_mode", "pd_joint_delta_pos")
    reward_mode = meta.get("reward_mode", "normalized_dense")
    pose_coef = float(meta.get("pose_reward_coef", 0.01))
    place_coef = float(meta.get("place_reward_coef", 0.1))

    # RT shaders are incompatible with visual obs_modes in ManiSkill 3, so we use obs_mode='none'.
    # The video is rendered through env.render() which uses the separate human-render camera.
    use_rt = shader[:2] == "rt"
    obs_mode = "none" if use_rt else "rgbd"
    if use_rt and sim_backend == "gpu":
        print("[rt-replay] WARNING: GPU sim is not supported with RT shaders in ManiSkill 3. Forcing sim_backend='cpu'.")
        sim_backend = "cpu"

    if camera == "eval":
        # Match the sensor camera the LightZero agent saw during eval (front view of the cube).
        # Note: human_render_camera_configs must be a Dict[str, dict] (ManiSkill calls .pop on values),
        # so we send a per-camera override dict, not a CameraConfig instance.
        pose = Pose.create(sapien_utils.look_at(eye=_EVAL_CAMERA_EYE, target=_EVAL_CAMERA_TARGET))
        human_render_camera_configs = {
            "render_camera": dict(
                pose=pose,
                width=image_size,
                height=image_size,
                fov=_EVAL_CAMERA_FOV,
                near=0.01,
                far=100,
            ),
        }
    else:
        # Use ManiSkill's stock side-angle render camera, just at the requested resolution.
        human_render_camera_configs = dict(width=image_size, height=image_size)

    env_kwargs = dict(
        pose_reward_coef=pose_coef,
        place_reward_coef=place_coef,
        obs_mode=obs_mode,
        control_mode=control_mode,
        render_mode="rgb_array",
        reward_mode=reward_mode,
        sensor_configs=dict(width=image_size, height=image_size),
        human_render_camera_configs=human_render_camera_configs,
        shader_dir=shader,
        enable_shadow=enable_shadow,
        sim_backend=sim_backend,
    )
    print(f"[rt-replay] Creating env {env_id} with shader='{shader}', obs_mode='{obs_mode}', sim_backend='{sim_backend}'")
    env = gymnasium.make(env_id, **env_kwargs)

    env.reset(seed=seed)
    env.action_space.seed(seed)

    frames: list[np.ndarray] = []
    frames.append(_frame_to_uint8_hwc(env.render()))

    for t, action in enumerate(actions):
        _obs, _r, terminated, truncated, _info = env.step(np.asarray(action, dtype=np.float32))
        frames.append(_frame_to_uint8_hwc(env.render()))
        if bool(terminated) or bool(truncated):
            print(f"[rt-replay] Episode finished early at step {t + 1}/{len(actions)} (terminated={bool(terminated)}, truncated={bool(truncated)})")
            break

    env.close()

    if save_mp4:
        os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
        h, w = frames[0].shape[:2]
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        writer = cv2.VideoWriter(out_path, fourcc, fps, (w, h))
        for frame in frames:
            writer.write(cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))
        writer.release()
        print(f"[rt-replay] Saved {len(frames)} frames → {out_path}")

    if save_frames:
        if frames_dir is None:
            frames_dir = os.path.splitext(out_path)[0] + "_frames"
        os.makedirs(frames_dir, exist_ok=True)
        ext = frames_format.lower()
        for i, frame in enumerate(frames):
            frame_path = os.path.join(frames_dir, f"frame_{i:04d}.{ext}")
            cv2.imwrite(frame_path, cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))
        print(f"[rt-replay] Saved {len(frames)} frames → {frames_dir}/frame_NNNN.{ext}")


def main() -> None:
    args = parse_args()
    trajectories = _resolve_trajectories(args.trajectory)

    if not args.save_mp4 and not args.save_frames:
        raise ValueError("Nothing to save: pass either mp4 (default) or --save-frames (or both).")

    out = Path(args.out)
    is_dir_out = out.suffix == "" or len(trajectories) > 1
    if is_dir_out:
        out.mkdir(parents=True, exist_ok=True)

    for traj in trajectories:
        stem = Path(traj).stem
        if is_dir_out:
            out_path = str(out / f"{stem}.{args.shader}.mp4")
            frames_dir = str(out / f"{stem}.{args.shader}_frames")
        else:
            out_path = str(out)
            frames_dir = os.path.splitext(out_path)[0] + "_frames"
        replay_one(
            npz_path=traj,
            out_path=out_path,
            shader=args.shader,
            image_size=args.image_size,
            fps=args.fps,
            sim_backend=args.sim_backend,
            enable_shadow=args.enable_shadow,
            camera=args.camera,
            save_mp4=args.save_mp4,
            save_frames=args.save_frames,
            frames_dir=frames_dir,
            frames_format=args.frames_format,
        )


if __name__ == "__main__":
    main()
