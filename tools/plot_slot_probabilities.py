import argparse
from pathlib import Path
from typing import Dict, List, Tuple

import cv2
import matplotlib.pyplot as plt
import numpy as np


def parse_causality_file(path: Path) -> Dict[Tuple[int, str], List[float]]:
    data: Dict[Tuple[int, str], List[float]] = {}
    with path.open("r", encoding="utf-8") as f:
        for line_idx, line in enumerate(f):
            line = line.strip()
            if not line:
                continue
            if line_idx == 0:
                # Header: step, type, slot_values
                continue
            parts = line.split("\t")
            if len(parts) != 3:
                raise ValueError(f"Invalid line format in {path}: {line}")
            step_str, prob_type, values_str = parts
            probs = [float(v) for v in values_str.split(",") if v]
            data[(int(step_str), prob_type)] = probs
    return data


def split_slot_strip(slot_strip_bgr: np.ndarray, num_slots: int) -> List[np.ndarray]:
    if slot_strip_bgr is None:
        raise ValueError("slot_strip_bgr is None")
    if slot_strip_bgr.ndim != 3:
        raise ValueError(f"Expected image with 3 dims (H, W, C), got shape {slot_strip_bgr.shape}")
    if num_slots <= 0:
        raise ValueError(f"num_slots must be positive, got {num_slots}")

    # np.array_split handles non-divisible widths safely.
    slot_chunks = np.array_split(slot_strip_bgr, num_slots, axis=1)
    return [cv2.cvtColor(chunk, cv2.COLOR_BGR2RGB) for chunk in slot_chunks]


def plot_probabilities_all_steps(
    probs_by_step: Dict[int, List[float]],
    slot_images_by_step: Dict[int, List[np.ndarray]],
    output_path: Path,
    slot_spacing: float,
) -> None:
    steps = sorted(probs_by_step.keys())
    if not steps:
        raise ValueError("No steps to plot.")

    num_slots = len(probs_by_step[steps[0]])
    for step in steps:
        if len(probs_by_step[step]) != num_slots:
            raise ValueError(f"Inconsistent number of slots at step {step}.")
        if len(slot_images_by_step[step]) != num_slots:
            raise ValueError(f"Inconsistent slot image count at step {step}.")

    global_max = max(max(v) for v in probs_by_step.values()) if probs_by_step else 1.0
    y_max = global_max * 1.15 if global_max > 0 else 1.0

    fig = plt.figure(figsize=(max(10, num_slots * 2.1), max(4.6, len(steps) * 4.8)))
    outer = fig.add_gridspec(len(steps), 1, hspace=0.33)

    for step_idx, step in enumerate(steps):
        inner = outer[step_idx].subgridspec(
            2,
            num_slots,
            height_ratios=[3.0, 2.0],
            hspace=0.03,
            wspace=slot_spacing,
        )

        probs = probs_by_step[step]
        slot_images = slot_images_by_step[step]

        for slot_idx, (slot_img, prob) in enumerate(zip(slot_images, probs)):
            top_ax = fig.add_subplot(inner[0, slot_idx])
            top_ax.imshow(slot_img)
            top_ax.axis("off")

            bar_ax = fig.add_subplot(inner[1, slot_idx])
            bar_ax.bar([0], [prob], width=0.62, color="#4E79A7")
            bar_ax.set_xlim(-0.8, 0.8)
            bar_ax.set_ylim(0, y_max)
            bar_ax.set_xticks([])
            bar_ax.set_yticks([])
            bar_ax.set_xlabel(f"slot {slot_idx}", fontsize=9)
            bar_ax.grid(False)
            for spine in bar_ax.spines.values():
                spine.set_visible(True)
                spine.set_linewidth(0.8)
                spine.set_color("#444444")
            bar_ax.text(0, prob + y_max * 0.02, f"{prob:.3f}", ha="center", va="bottom", fontsize=8)

        # Keep heading close to images by using a very small offset from the block's top.
        step_bbox = outer[step_idx].get_position(fig)
        fig.text(0.5, step_bbox.y1 + 0.002, f"Step {step}", ha="center", va="bottom", fontsize=13)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Build slot-wise probability visualizations for all steps."
    )
    parser.add_argument(
        "--causality-file",
        type=Path,
        default=Path("visuals/oz_policy_log/causal_probs.txt"),
        help="Path to causality text file.",
    )
    parser.add_argument(
        "--slot-dir",
        type=Path,
        default=Path("visuals/only_on_slot_recon"),
        help="Directory with slot strip images named like sa_recon_0000.jpg.",
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=Path("visuals/slot_probability_plots"),
        help="Output directory for generated figures.",
    )
    parser.add_argument(
        "--slot-spacing",
        type=float,
        default=0.05,
        help="Horizontal spacing between slot columns in the plot.",
    )
    args = parser.parse_args()

    causality = parse_causality_file(args.causality_file)

    steps = sorted({step for step, prob_type in causality.keys() if prob_type == "policy"})
    if not steps:
        raise ValueError(f"No policy entries found in {args.causality_file}")

    policy_by_step: Dict[int, List[float]] = {}
    value_by_step: Dict[int, List[float]] = {}
    slot_images_by_step: Dict[int, List[np.ndarray]] = {}

    for step in steps:
        policy_probs = causality.get((step, "policy"))
        value_probs = causality.get((step, "value"))
        if policy_probs is None or value_probs is None:
            raise KeyError(f"Expected both 'policy' and 'value' for step {step} in {args.causality_file}")
        if len(policy_probs) != len(value_probs):
            raise ValueError(
                f"policy/value slot count mismatch at step {step}: {len(policy_probs)} vs {len(value_probs)}"
            )

        slot_path = args.slot_dir / f"sa_recon_{step:04d}.jpg"
        slot_strip = cv2.imread(str(slot_path), cv2.IMREAD_COLOR)
        if slot_strip is None:
            raise FileNotFoundError(f"Slot image not found or unreadable: {slot_path}")

        slot_images = split_slot_strip(slot_strip, len(policy_probs))

        policy_by_step[step] = policy_probs
        value_by_step[step] = value_probs
        slot_images_by_step[step] = slot_images

    policy_out = args.out_dir / "slot_probs_policy_all_steps.png"
    value_out = args.out_dir / "slot_probs_value_all_steps.png"

    plot_probabilities_all_steps(
        probs_by_step=policy_by_step,
        slot_images_by_step=slot_images_by_step,
        output_path=policy_out,
        slot_spacing=args.slot_spacing,
    )
    plot_probabilities_all_steps(
        probs_by_step=value_by_step,
        slot_images_by_step=slot_images_by_step,
        output_path=value_out,
        slot_spacing=args.slot_spacing,
    )

    print(f"Saved: {policy_out}")
    print(f"Saved: {value_out}")


if __name__ == "__main__":
    main()
