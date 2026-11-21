"""Export overlays of three Lift frames with robosuite ground-truth segmentation.

Run from the repository root:
    MUJOCO_GL=egl /home/rodya-rad/miniconda3/envs/lightzero/bin/python \
        -m zoo.robosuite.entry.export_gt_masks --output-dir /tmp/opencode/robosuite_overlays
"""

import argparse
from pathlib import Path

import numpy as np
import robosuite
from PIL import Image, ImageDraw
from robosuite import load_controller_config

from zoo.ocr.dino_object.segmentation import ROBOSUITE_NAMES, robosuite_own_seg_v1
from zoo.robosuite.env.robosuite import FixedPositionSampler


def save_overlay(observation, env, timestep, output_dir, segmentation):
    # Both sensors use the same camera and image convention. Match the vertical
    # flip used by the repository's RobosuiteEnv image preprocessing.
    rgb = np.flipud(observation["frontview_image"])
    sensor_type = "element" if segmentation == "own_seg_v1" else segmentation
    labels = np.flipud(observation[f"frontview_segmentation_{sensor_type}"][..., 0])
    if segmentation == "own_seg_v1":
        labels = robosuite_own_seg_v1(labels, env)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Class/instance masks reserve 0 for background; element masks use -1.
    foreground = labels >= 0 if segmentation == "element" else labels > 0
    overlay = rgb.copy()
    visible_ids = np.unique(labels[foreground])
    if segmentation == "own_seg_v1":
        names = ROBOSUITE_NAMES
    elif segmentation == "instance":
        names = list(env.model.instances_to_ids)
    elif segmentation == "class":
        names = list(env.model.classes_to_ids)
    else:
        names = None
    legend = []
    for label_id in visible_ids:
        mask = labels == label_id
        color = np.array(
            [(37 * int(label_id) + 67) % 256,
             (97 * int(label_id) + 113) % 256,
             (173 * int(label_id) + 47) % 256],
            dtype=np.uint8,
        )
        overlay[mask] = (0.35 * rgb[mask] + 0.65 * color).astype(np.uint8)
        if names is not None:
            # Bright one-pixel contours keep small objects (such as the cube)
            # visible even when the overlay is shown at its original size.
            interior = mask.copy()
            interior[1:, :] &= mask[:-1, :]
            interior[:-1, :] &= mask[1:, :]
            interior[:, 1:] &= mask[:, :-1]
            interior[:, :-1] &= mask[:, 1:]
            overlay[mask & ~interior] = (255, 255, 255)
            legend.append((color, f"{names[int(label_id) - 1]}: {int(mask.sum())} px"))

    lines = len(legend) if legend else 1
    canvas = Image.new("RGB", (rgb.shape[1], rgb.shape[0] + lines * 18 + 8), (30, 30, 30))
    canvas.paste(Image.fromarray(overlay), (0, 0))
    draw = ImageDraw.Draw(canvas)
    if legend:
        for index, (color, text) in enumerate(legend):
            y = rgb.shape[0] + 4 + index * 18
            draw.rectangle((4, y + 2, 14, y + 12), fill=tuple(int(x) for x in color))
            draw.text((20, y), text, fill="white")
    else:
        draw.text((4, rgb.shape[0] + 4), f"{len(visible_ids)} visible geoms", fill="white")
    path = output_dir / f"overlay_step_{timestep:02d}.png"
    canvas.save(path)
    print(f"Saved {path} ({len(visible_ids)} visible labels)")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=Path("/tmp/opencode/robosuite_overlays"))
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--segmentation", choices=("instance", "class", "element", "own_seg_v1"), default="instance")
    args = parser.parse_args()
    if args.segmentation == "own_seg_v1" and args.output_dir == Path("/tmp/opencode/robosuite_overlays"):
        args.output_dir = Path("/tmp/opencode/robosuite_own_seg_v1")

    np.random.seed(args.seed)
    env = robosuite.make(
        "Lift",
        robots=["Panda"],
        gripper_types="default",
        controller_configs=load_controller_config(default_controller="OSC_POSITION"),
        env_configuration="default",
        use_camera_obs=True,
        use_object_obs=False,
        reward_shaping=True,
        has_renderer=False,
        has_offscreen_renderer=True,
        control_freq=20,
        horizon=125,
        camera_names="frontview",
        camera_heights=224,
        camera_widths=224,
        camera_segmentations="element" if args.segmentation == "own_seg_v1" else args.segmentation,
        placement_initializer=FixedPositionSampler("ObjectSampler", "Lift"),
        initialization_noise={"magnitude": 0.5, "type": "uniform"},
        ignore_done=True,
    )
    try:
        env.reset()
        low, high = env.action_spec
        for timestep in range(1, 4):
            action = np.random.uniform(low, high).astype(np.float32)
            observation, _, _, _ = env.step(action)
            save_overlay(observation, env, timestep, args.output_dir, args.segmentation)
    finally:
        env.close()


if __name__ == "__main__":
    main()
