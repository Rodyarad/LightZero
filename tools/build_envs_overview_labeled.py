import argparse
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build a 2x4 environments overview image with labels."
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
        default=Path("visuals/envs_labeled.jpg"),
        help="Output collage image path.",
    )
    parser.add_argument(
        "--video-frame-glob",
        type=str,
        default="frame_*.jpg",
        help="Glob for frame images inside video_frames.",
    )
    parser.add_argument(
        "--tile-size",
        type=int,
        default=280,
        help="Square size for each environment image tile.",
    )
    parser.add_argument(
        "--font-path",
        type=Path,
        default=None,
        help="Optional path to a .ttf font file (e.g., Roboto-Regular.ttf).",
    )
    return parser.parse_args()


def _first_match(directory: Path, glob_pattern: str) -> Optional[Path]:
    matches = sorted(directory.glob(glob_pattern))
    if matches:
        return matches[0]
    return None


def _load_bgr(path: Path) -> Optional[np.ndarray]:
    return cv2.imread(str(path), cv2.IMREAD_COLOR)


def _resize_square(img: np.ndarray, size: int) -> np.ndarray:
    return cv2.resize(img, (size, size), interpolation=cv2.INTER_AREA)


def _find_roboto_font(user_font_path: Optional[Path]) -> Optional[Path]:
    if user_font_path is not None and user_font_path.exists():
        return user_font_path

    candidates = [
        Path("/usr/share/fonts/truetype/roboto/Roboto-Regular.ttf"),
        Path("/usr/share/fonts/truetype/roboto/unhinted/RobotoTTF/Roboto-Regular.ttf"),
        Path("/usr/share/fonts/truetype/google/Roboto-Regular.ttf"),
    ]
    for path in candidates:
        if path.exists():
            return path
    return None


def _draw_label(
    canvas: np.ndarray,
    text: str,
    x_center: int,
    y_top: int,
    max_width: int,
    font_path: Optional[Path],
    font_scale: float = 0.9,
) -> None:
    if font_path is not None:
        font_size = max(12, int(round(36 * font_scale)))
        font = ImageFont.truetype(str(font_path), font_size)
        probe = ImageDraw.Draw(Image.new("RGB", (1, 1), (255, 255, 255)))
        bbox = probe.textbbox((0, 0), text, font=font)
        text_w = bbox[2] - bbox[0]
        text_h = bbox[3] - bbox[1]

        while text_w > max_width and font_size > 10:
            font_size -= 1
            font = ImageFont.truetype(str(font_path), font_size)
            bbox = probe.textbbox((0, 0), text, font=font)
            text_w = bbox[2] - bbox[0]
            text_h = bbox[3] - bbox[1]

        x = x_center - (text_w // 2)
        y = y_top + text_h

        pil_img = Image.fromarray(cv2.cvtColor(canvas, cv2.COLOR_BGR2RGB))
        draw = ImageDraw.Draw(pil_img)
        draw.text((x, y - text_h), text, font=font, fill=(20, 20, 20))
        canvas[:, :, :] = cv2.cvtColor(np.array(pil_img), cv2.COLOR_RGB2BGR)
        return

    # Fallback to OpenCV built-in font when Roboto is unavailable.
    font = cv2.FONT_HERSHEY_SIMPLEX
    thickness = 2
    text_size, _ = cv2.getTextSize(text, font, font_scale, thickness)
    text_w, text_h = text_size

    while text_w > max_width and font_scale > 0.35:
        font_scale -= 0.05
        text_size, _ = cv2.getTextSize(text, font, font_scale, thickness)
        text_w, text_h = text_size

    x = x_center - (text_w // 2)
    y = y_top + text_h
    cv2.putText(canvas, text, (x, y), font, font_scale, (20, 20, 20), thickness, cv2.LINE_AA)


def main() -> None:
    args = parse_args()
    root_dir = args.root_dir.resolve()
    tile_size = int(args.tile_size)
    font_path = _find_roboto_font(args.font_path)

    env_layout: List[List[Tuple[str, str]]] = [
        [
            ("Object Goal", "visuals_goal"),
            ("Object Interaction", "visuals_inter"),
            ("Property Comparison", "visuals_oddprop"),
            ("Object Comparison", "visuals_oddobj"),
        ],
        [
            ("Block Lifting", "visuals_robo"),
            ("Cube Pushing", "visuals_mani"),
            ("Object Reaching", "visuals_cw"),
            ("Defend The Line", "visuals_viz"),
        ],
    ]

    # Validate all required directories and frame files first.
    images: Dict[str, np.ndarray] = {}
    missing: List[str] = []
    for row in env_layout:
        for label, dirname in row:
            frame_dir = root_dir / dirname / "video_frames"
            if not frame_dir.exists():
                missing.append(f"{dirname}: missing video_frames")
                continue
            frame_path = _first_match(frame_dir, args.video_frame_glob)
            if frame_path is None:
                missing.append(f"{dirname}: no frame matching {args.video_frame_glob}")
                continue
            frame_bgr = _load_bgr(frame_path)
            if frame_bgr is None:
                missing.append(f"{dirname}: failed to read {frame_path.name}")
                continue
            images[label] = _resize_square(frame_bgr, tile_size)

    if missing:
        raise RuntimeError("Cannot build overview:\n" + "\n".join(f"- {msg}" for msg in missing))

    cols = 4
    rows = 2
    gap_x = 96
    gap_y = 112
    outer_pad = 56
    label_h = 44

    canvas_w = outer_pad * 2 + cols * tile_size + (cols - 1) * gap_x
    canvas_h = outer_pad * 2 + rows * (label_h + tile_size) + (rows - 1) * gap_y
    canvas = np.full((canvas_h, canvas_w, 3), 255, dtype=np.uint8)

    for r in range(rows):
        row_top = outer_pad + r * (label_h + tile_size + gap_y)
        img_top = row_top + label_h
        for c in range(cols):
            label, _ = env_layout[r][c]
            x_left = outer_pad + c * (tile_size + gap_x)
            x_center = x_left + tile_size // 2
            tile = images[label]
            canvas[img_top : img_top + tile_size, x_left : x_left + tile_size] = tile
            _draw_label(
                canvas=canvas,
                text=label,
                x_center=x_center,
                y_top=row_top,
                max_width=tile_size - 8,
                font_path=font_path,
            )

    args.output_path.parent.mkdir(parents=True, exist_ok=True)
    ok = cv2.imwrite(str(args.output_path), canvas)
    if not ok:
        raise RuntimeError(f"Failed to write output image: {args.output_path}")

    print(f"Saved overview to {args.output_path}")
    if font_path is not None:
        print(f"Using font: {font_path}")
    else:
        print("Roboto font not found, fallback to OpenCV font.")


if __name__ == "__main__":
    main()
