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


def plot_probabilities_for_step(
    step: int,
    probs: List[float],
    slot_images: List[np.ndarray],
    output_path: Path,
    slot_spacing: float,
    highlight_slots: List[int],
    highlight_color_map: Dict[int, str],
    viz_mode: int,
) -> None:
    if len(probs) != len(slot_images):
        raise ValueError(f"Slot/probability size mismatch at step {step}: {len(slot_images)} vs {len(probs)}")

    num_slots = len(probs)
    y_max = (max(probs) * 1.15) if len(probs) > 0 and max(probs) > 0 else 1.0

    if viz_mode == 1:
        fig_height = 4.8
    else:
        fig_height = 3.6
    fig = plt.figure(figsize=(max(10, num_slots * 2.1), fig_height))
    inner = fig.add_gridspec(
        2,
        num_slots,
        height_ratios=_height_ratios(viz_mode),
        hspace=0.03,
        wspace=slot_spacing,
    )

    _draw_step(
        fig=fig,
        inner=inner,
        probs=probs,
        slot_images=slot_images,
        y_max=y_max,
        highlight_slots=highlight_slots,
        highlight_color_map=highlight_color_map,
        viz_mode=viz_mode,
    )

    fig.suptitle(f"Step {step}", fontsize=13, y=1.08, fontweight="bold")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close(fig)


def _height_ratios(viz_mode: int) -> List[float]:
    if viz_mode == 1:
        return [3.0, 2.0]
    # In numeric mode keep the number strip compact.
    return [2.0, 0.55]


def _draw_step(
    fig: plt.Figure,
    inner,
    probs: List[float],
    slot_images: List[np.ndarray],
    y_max: float,
    highlight_slots: List[int],
    highlight_color_map: Dict[int, str],
    viz_mode: int,
) -> None:
    for slot_idx, (slot_img, prob) in enumerate(zip(slot_images, probs)):
        top_ax = fig.add_subplot(inner[0, slot_idx])
        top_ax.imshow(slot_img)
        top_ax.set_xticks([])
        top_ax.set_yticks([])
        if viz_mode == 2:
            top_ax.text(
                0.5,
                1.06,
                f"{prob:.3f}",
                ha="center",
                va="bottom",
                fontsize=13,
                fontweight="bold",
                transform=top_ax.transAxes,
                bbox=dict(facecolor="white", edgecolor="none", alpha=0.85, boxstyle="round,pad=0.15"),
            )
        for spine_name, spine in top_ax.spines.items():
            if slot_idx in highlight_slots:
                spine.set_visible(True)
                spine.set_color(highlight_color_map.get(slot_idx, "red"))
                spine.set_linewidth(2.5)
            else:
                spine.set_visible(False)

        bar_ax = fig.add_subplot(inner[1, slot_idx])
        if viz_mode == 1:
            bar_ax.bar([0], [prob], width=0.62, color="#4E79A7")
            bar_ax.set_xlim(-0.8, 0.8)
            bar_ax.set_ylim(0, y_max)
            bar_ax.set_xticks([])
            bar_ax.set_yticks([])
            bar_ax.grid(False)
            for spine in bar_ax.spines.values():
                spine.set_visible(True)
                spine.set_linewidth(0.8)
                spine.set_color("#444444")
            bar_ax.text(0, prob + y_max * 0.02, f"{prob:.3f}", ha="center", va="bottom", fontsize=8)
        else:
            bar_ax.set_xticks([])
            bar_ax.set_yticks([])
            bar_ax.set_frame_on(False)


def plot_probabilities_selected_steps(
    probs_by_step: Dict[int, List[float]],
    slot_images_by_step: Dict[int, List[np.ndarray]],
    selected_steps: List[int],
    output_path: Path,
    slot_spacing: float,
    highlight_slots: List[int],
    highlight_color_map: Dict[int, str],
    viz_mode: int,
) -> None:
    if len(selected_steps) == 0:
        return
    valid_steps = sorted(selected_steps)
    num_slots = len(probs_by_step[valid_steps[0]])
    for step in valid_steps:
        if step not in probs_by_step:
            raise ValueError(f"Requested selected step {step} is missing in probability data.")
        if step not in slot_images_by_step:
            raise ValueError(f"Requested selected step {step} is missing in slot image data.")
        if len(probs_by_step[step]) != num_slots:
            raise ValueError(f"Inconsistent number of slots at step {step}.")
        if len(slot_images_by_step[step]) != num_slots:
            raise ValueError(f"Inconsistent slot image count at step {step}.")

    base_height = 4.8 if viz_mode == 1 else 3.6
    fig = plt.figure(figsize=(max(10, num_slots * 2.1), max(base_height, base_height * len(valid_steps))))
    outer = fig.add_gridspec(len(valid_steps), 1, hspace=0.33)

    for row_idx, step in enumerate(valid_steps):
        probs = probs_by_step[step]
        slot_images = slot_images_by_step[step]
        y_max = (max(probs) * 1.15) if len(probs) > 0 and max(probs) > 0 else 1.0
        inner = outer[row_idx].subgridspec(
            2,
            num_slots,
            height_ratios=_height_ratios(viz_mode),
            hspace=0.03,
            wspace=slot_spacing,
        )
        _draw_step(
            fig=fig,
            inner=inner,
            probs=probs,
            slot_images=slot_images,
            y_max=y_max,
            highlight_slots=highlight_slots,
            highlight_color_map=highlight_color_map,
            viz_mode=viz_mode,
        )
        step_bbox = outer[row_idx].get_position(fig)
        fig.text(0.5, step_bbox.y1 + 0.03, f"Step {step}", ha="center", va="bottom", fontsize=13, fontweight="bold")

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
        default=Path("visuals/sa_slots_get_samples"),
        help="Directory with slot strip images.",
    )
    parser.add_argument(
        "--slot-filename-template",
        type=str,
        default="slots_sample_{step:04d}.jpg",
        help="Template for slot image files in slot-dir.",
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
    parser.add_argument(
        "--highlight-slots",
        type=int,
        nargs="+",
        default=[],
        help="Slot indices to highlight with red border (e.g. --highlight-slots 1 2).",
    )
    parser.add_argument(
        "--highlight-colors",
        type=int,
        nargs="+",
        default=[],
        help=(
            "Per-highlight slot color codes aligned with --highlight-slots: "
            "1=red, 0=blue, 2=green. Example: --highlight-slots 2 4 5 --highlight-colors 1 0 2"
        ),
    )
    parser.add_argument(
        "--viz-mode",
        type=int,
        choices=[1, 2],
        default=2,
        help="1 = bars, 2 = numeric values under slots.",
    )
    parser.add_argument(
        "--merge-steps",
        type=int,
        nargs="+",
        default=[0, 3, 7],
        help="Optional list of step indices to merge into one figure (e.g. --merge-steps 0 2 5).",
    )
    args = parser.parse_args()
    if args.highlight_colors is None:
        args.highlight_colors = [1] * len(args.highlight_slots)
    if len(args.highlight_colors) != len(args.highlight_slots):
        raise ValueError(
            "--highlight-colors must have the same length as --highlight-slots. "
            f"Got {len(args.highlight_colors)} vs {len(args.highlight_slots)}."
        )
    valid_color_codes = {0, 1, 2}
    invalid_codes = sorted({code for code in args.highlight_colors if code not in valid_color_codes})
    if invalid_codes:
        raise ValueError(
            f"Unsupported highlight color code(s): {invalid_codes}. "
            "Allowed values are 0 (blue), 1 (red), 2 (green)."
        )
    color_code_to_name = {0: "blue", 1: "red", 2: "green"}
    highlight_color_map = {
        slot_idx: color_code_to_name[color_code]
        for slot_idx, color_code in zip(args.highlight_slots, args.highlight_colors)
    }

    causality = parse_causality_file(args.causality_file)

    steps = sorted({step for step, prob_type in causality.keys() if prob_type == "policy"})
    if not steps:
        raise ValueError(f"No policy entries found in {args.causality_file}")

    saved_paths: List[Path] = []
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

        slot_path = args.slot_dir / args.slot_filename_template.format(step=step)
        slot_strip = cv2.imread(str(slot_path), cv2.IMREAD_COLOR)
        if slot_strip is None:
            raise FileNotFoundError(f"Slot image not found or unreadable: {slot_path}")

        slot_images = split_slot_strip(slot_strip, len(policy_probs))
        policy_by_step[step] = policy_probs
        value_by_step[step] = value_probs
        slot_images_by_step[step] = slot_images

        policy_out = args.out_dir / f"slot_probs_policy_step_{step:04d}.png"
        value_out = args.out_dir / f"slot_probs_value_step_{step:04d}.png"

        plot_probabilities_for_step(
            step=step,
            probs=policy_probs,
            slot_images=slot_images,
            output_path=policy_out,
            slot_spacing=args.slot_spacing,
            highlight_slots=args.highlight_slots,
            highlight_color_map=highlight_color_map,
            viz_mode=args.viz_mode,
        )
        plot_probabilities_for_step(
            step=step,
            probs=value_probs,
            slot_images=slot_images,
            output_path=value_out,
            slot_spacing=args.slot_spacing,
            highlight_slots=args.highlight_slots,
            highlight_color_map=highlight_color_map,
            viz_mode=args.viz_mode,
        )
        saved_paths.extend([policy_out, value_out])

    if args.merge_steps is not None and len(args.merge_steps) > 0:
        merged_policy_out = args.out_dir / "slot_probs_policy_selected_steps.png"
        merged_value_out = args.out_dir / "slot_probs_value_selected_steps.png"
        plot_probabilities_selected_steps(
            probs_by_step=policy_by_step,
            slot_images_by_step=slot_images_by_step,
            selected_steps=args.merge_steps,
            output_path=merged_policy_out,
            slot_spacing=args.slot_spacing,
            highlight_slots=args.highlight_slots,
            highlight_color_map=highlight_color_map,
            viz_mode=args.viz_mode,
        )
        plot_probabilities_selected_steps(
            probs_by_step=value_by_step,
            slot_images_by_step=slot_images_by_step,
            selected_steps=args.merge_steps,
            output_path=merged_value_out,
            slot_spacing=args.slot_spacing,
            highlight_slots=args.highlight_slots,
            highlight_color_map=highlight_color_map,
            viz_mode=args.viz_mode,
        )
        saved_paths.extend([merged_policy_out, merged_value_out])

    print(f"Saved {len(saved_paths)} files to {args.out_dir}")


if __name__ == "__main__":
    main()
