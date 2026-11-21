"""Visualize three PushCubeCustom-v1 steps with grouped ground-truth masks.

From the repository root:
    /home/rodya-rad/miniconda3/envs/lightzero/bin/python \
        -m zoo.maniskill.entry.export_gt_masks
"""

import argparse
from pathlib import Path

import gymnasium as gym
import numpy as np
from mani_skill.utils import common
from PIL import Image, ImageDraw

from zoo.ocr.dino_object.segmentation import MANISKILL_NAMES, maniskill_own_seg_v1
# Import registers the same custom task that the LightZero ManiSkill wrapper uses.
from zoo.maniskill.env.maniskill3 import PushCubeCustomEnv  # noqa: F401


def save_overlay(observation, id_map, timestep, output_dir):
    camera = observation["sensor_data"]["base_camera"]
    rgb = common.to_numpy(camera["rgb"][0])
    raw_labels = common.to_numpy(camera["segmentation"][0, ..., 0])
    labels = maniskill_own_seg_v1(raw_labels, id_map)

    overlay = rgb.copy()
    legend = []
    for group, name in enumerate(MANISKILL_NAMES, start=1):
        mask = labels == group
        color = np.array(
            [(37 * group + 67) % 256, (97 * group + 113) % 256, (173 * group + 47) % 256],
            dtype=np.uint8,
        )
        overlay[mask] = (0.35 * rgb[mask] + 0.65 * color).astype(np.uint8)
        interior = mask.copy()
        interior[1:, :] &= mask[:-1, :]
        interior[:-1, :] &= mask[1:, :]
        interior[:, 1:] &= mask[:, :-1]
        interior[:, :-1] &= mask[:, 1:]
        overlay[mask & ~interior] = (255, 255, 255)
        legend.append((color, f"{name}: {int(mask.sum())} px"))

    height, width = rgb.shape[:2]
    canvas = Image.new("RGB", (width, height + len(legend) * 18 + 8), (30, 30, 30))
    canvas.paste(Image.fromarray(overlay), (0, 0))
    draw = ImageDraw.Draw(canvas)
    for index, (color, text) in enumerate(legend):
        y = height + 4 + index * 18
        draw.rectangle((4, y + 2, 14, y + 12), fill=tuple(int(x) for x in color))
        draw.text((20, y), text, fill="white")

    output_dir.mkdir(parents=True, exist_ok=True)
    path = output_dir / f"overlay_step_{timestep:02d}.png"
    canvas.save(path)
    print(f"Saved {path}")
    for _, text in legend:
        print(f"  {text}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=Path("/tmp/opencode/maniskill_own_seg_v1"))
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    env = gym.make(
        "PushCubeCustom-v1",
        obs_mode="rgbd",
        control_mode="pd_joint_delta_pos",
        render_mode="rgb_array",
        reward_mode="dense",
        sensor_configs=dict(width=224, height=224),
    )
    try:
        env.reset(seed=args.seed)
        env.action_space.seed(args.seed)
        id_map = env.unwrapped.segmentation_id_map
        for timestep in range(1, 4):
            observation, _, _, _, _ = env.step(env.action_space.sample())
            save_overlay(observation, id_map, timestep, args.output_dir)
    finally:
        env.close()


if __name__ == "__main__":
    main()
