import argparse
from pathlib import Path
from typing import List, Optional, Tuple

import cv2
import numpy as np

EXCLUDED_VISUAL_DIRS = {"visuals_oddprop", "visuals_oddobj", "visuals_inter"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Build a 2-column overview image from visuals_* directories: "
            "left=first video frame, right=first get_samples image."
        )
    )
    parser.add_argument(
        "--root-dir",
        type=Path,
        default=Path("."),
        help="Project root directory containing visuals_* folders.",
    )
    parser.add_argument(
        "--output-path",
        type=Path,
        default=Path("visuals/envs.jpg"),
        help="Output collage image path.",
    )
    parser.add_argument(
        "--video-frame-glob",
        type=str,
        default="frame_*.jpg",
        help="Glob for frame images inside video_frames.",
    )
    parser.add_argument(
        "--sample-globs",
        type=str,
        nargs="+",
        default=["slots_sample_*.jpg", "slots_sample_*.png"],
        help="Ordered globs for get_samples images inside sa_slots_get_samples.",
    )
    parser.add_argument("--tile-size", type=int, default=64, help="Tile size for frame and each slot image.")
    return parser.parse_args()


def _first_match(directory: Path, globs: List[str]) -> Optional[Path]:
    for g in globs:
        matches = sorted(directory.glob(g))
        if matches:
            return matches[0]
    return None


def _load_bgr(path: Path) -> Optional[np.ndarray]:
    img = cv2.imread(str(path), cv2.IMREAD_COLOR)
    return img


def _resize_square(img: np.ndarray, size: int) -> np.ndarray:
    return cv2.resize(img, (size, size), interpolation=cv2.INTER_NEAREST)


def _sample_strip_to_slots_row(sample_bgr: np.ndarray, tile_size: int) -> np.ndarray:
    h, w = sample_bgr.shape[:2]
    # Slot strips are horizontal concatenations of square-like slot tiles.
    num_slots = max(1, int(round(w / float(max(h, 1)))))
    chunks = np.array_split(sample_bgr, num_slots, axis=1)
    slot_tiles = [_resize_square(c, tile_size) for c in chunks]
    slot_gap = 4
    row_w = sum(t.shape[1] for t in slot_tiles) + slot_gap * (len(slot_tiles) - 1)
    row = np.full((tile_size, row_w, 3), 255, dtype=np.uint8)
    x = 0
    for i, tile in enumerate(slot_tiles):
        row[:, x : x + tile.shape[1]] = tile
        x += tile.shape[1]
        if i < len(slot_tiles) - 1:
            x += slot_gap
    return row


def _build_row(frame_bgr: np.ndarray, slots_row_bgr: np.ndarray, tile_size: int, col0_w: int, col1_w: int) -> np.ndarray:
    frame_tile = _resize_square(frame_bgr, tile_size)
    sample_tile = slots_row_bgr
    gap = 12
    side_pad = 8
    row_w = side_pad + col0_w + gap + col1_w + side_pad
    row = np.full((tile_size + 14, row_w, 3), 255, dtype=np.uint8)

    y0 = 7
    col0_x = side_pad
    col1_x = side_pad + col0_w + gap

    # Center each tile inside its own column.
    frame_x = col0_x + (col0_w - frame_tile.shape[1]) // 2
    sample_x = col1_x + (col1_w - sample_tile.shape[1]) // 2

    row[y0 : y0 + tile_size, frame_x : frame_x + frame_tile.shape[1]] = frame_tile
    row[y0 : y0 + tile_size, sample_x : sample_x + sample_tile.shape[1]] = sample_tile
    return row


def main() -> None:
    args = parse_args()
    root_dir = args.root_dir.resolve()

    visual_dirs = sorted(
        p for p in root_dir.iterdir() if p.is_dir() and p.name.startswith("visuals_") and p.name not in EXCLUDED_VISUAL_DIRS
    )
    if not visual_dirs:
        raise FileNotFoundError(f"No visuals_* directories found in {root_dir}")

    valid_pairs: List[Tuple[np.ndarray, np.ndarray]] = []
    skipped: List[Tuple[str, str]] = []
    for vd in visual_dirs:
        frame_dir = vd / "video_frames"
        sample_dir = vd / "sa_slots_get_samples"
        frame_path = _first_match(frame_dir, [args.video_frame_glob]) if frame_dir.exists() else None
        sample_path = _first_match(sample_dir, args.sample_globs) if sample_dir.exists() else None
        if frame_path is None or sample_path is None:
            reason = []
            if frame_path is None:
                reason.append("missing first video frame")
            if sample_path is None:
                reason.append("missing first get_samples image")
            skipped.append((vd.name, ", ".join(reason)))
            continue

        frame_bgr = _load_bgr(frame_path)
        sample_bgr = _load_bgr(sample_path)
        if frame_bgr is None or sample_bgr is None:
            skipped.append((vd.name, "failed to read image files"))
            continue

        valid_pairs.append((frame_bgr, sample_bgr))

    if not valid_pairs:
        raise RuntimeError("No valid visuals_* directories with both frame and get_samples images.")

    tile_size = int(args.tile_size)
    sample_rows = [_sample_strip_to_slots_row(sample_bgr, tile_size) for _, sample_bgr in valid_pairs]
    col0_w = tile_size
    col1_w = max(r.shape[1] for r in sample_rows)
    rows = [
        _build_row(frame_bgr, sample_rows[i], tile_size=tile_size, col0_w=col0_w, col1_w=col1_w)
        for i, (frame_bgr, _) in enumerate(valid_pairs)
    ]

    gap_h = 12
    canvas_h = sum(r.shape[0] for r in rows) + gap_h * (len(rows) - 1)
    canvas_w = max(r.shape[1] for r in rows)
    canvas = np.full((canvas_h, canvas_w, 3), 255, dtype=np.uint8)
    y = 0
    for i, row in enumerate(rows):
        canvas[y : y + row.shape[0], : row.shape[1]] = row
        y += row.shape[0]
        if i < len(rows) - 1:
            y += gap_h

    args.output_path.parent.mkdir(parents=True, exist_ok=True)
    ok = cv2.imwrite(str(args.output_path), canvas)
    if not ok:
        raise RuntimeError(f"Failed to write output image: {args.output_path}")

    print(f"Saved overview to {args.output_path}")
    if skipped:
        print("Skipped directories:")
        for name, reason in skipped:
            print(f"  - {name}: {reason}")


if __name__ == "__main__":
    main()
