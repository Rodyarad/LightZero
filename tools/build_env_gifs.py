import argparse
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

from PIL import Image, ImageDraw, ImageFont


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Build improved per-environment GIFs and a labeled 2x4 grid GIF "
            "from visuals_*/video_frames."
        )
    )
    parser.add_argument(
        "--root-dir",
        type=Path,
        default=Path("."),
        help="Project root directory containing visuals_* folders.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("visuals/gifs"),
        help="Directory where generated GIFs are stored (including grid GIF).",
    )
    parser.add_argument(
        "--frame-glob",
        type=str,
        default="frame_*.jpg",
        help="Glob pattern for frame files inside video_frames.",
    )
    parser.add_argument(
        "--fps",
        type=float,
        default=8.0,
        help="Frames per second for GIF playback.",
    )
    parser.add_argument(
        "--max-frames",
        type=int,
        default=0,
        help="Optional cap on frame count per GIF. 0 means use all frames.",
    )
    parser.add_argument(
        "--tile-size",
        type=int,
        default=0,
        help="Square resize size for individual GIFs. 0 keeps original frame size.",
    )
    parser.add_argument(
        "--grid-tile-size",
        type=int,
        default=360,
        help="Square tile size used for the combined 2x4 grid GIF.",
    )
    parser.add_argument(
        "--font-path",
        type=Path,
        default=None,
        help="Optional path to Roboto .ttf. If omitted, common system paths are checked.",
    )
    parser.add_argument(
        "--grid-output-name",
        type=str,
        default="envs_grid",
        help="Filename for the combined labeled 2x4 animation (extension added automatically).",
    )
    parser.add_argument(
        "--skip-individual",
        action="store_true",
        help="Do not generate per-environment GIF files.",
    )
    parser.add_argument(
        "--skip-grid",
        action="store_true",
        help="Do not generate the combined grid GIF.",
    )
    parser.add_argument(
        "--loop",
        type=int,
        default=0,
        help="Animation loop count. 0 means infinite looping.",
    )
    parser.add_argument(
        "--dither",
        type=str,
        choices=["floyd", "none"],
        default="floyd",
        help="Dithering mode for GIF quantization.",
    )
    parser.add_argument(
        "--output-format",
        type=str,
        choices=["webp", "gif"],
        default="webp",
        help="Animation output format. webp is recommended for quality.",
    )
    parser.add_argument(
        "--webp-mode",
        type=str,
        choices=["lossless", "lossy"],
        default="lossless",
        help="WEBP compression mode.",
    )
    parser.add_argument(
        "--webp-quality",
        type=int,
        default=95,
        help="WEBP quality for lossy mode (0-100).",
    )
    return parser.parse_args()


def _iter_envs() -> Sequence[Tuple[str, str, str]]:
    return [
        ("Object Goal", "object_goal", "visuals_goal"),
        ("Object Interaction", "object_interaction", "visuals_inter"),
        ("Property Comparison", "property_comparison", "visuals_oddprop"),
        ("Object Comparison", "object_comparison", "visuals_oddobj"),
        ("Block Lifting", "block_lifting", "visuals_robo"),
        ("Cube Pushing", "cube_pushing", "visuals_mani"),
        ("Object Reaching", "object_reaching", "visuals_cw"),
        ("Defend The Line", "defend_the_line", "visuals_viz"),
    ]


def _collect_frames(frames_dir: Path, frame_glob: str, max_frames: int) -> List[Path]:
    frames = sorted(frames_dir.glob(frame_glob))
    if max_frames > 0:
        frames = frames[:max_frames]
    return frames


def _load_frames(frame_paths: Sequence[Path], tile_size: int) -> List[Image.Image]:
    images: List[Image.Image] = []
    for p in frame_paths:
        with Image.open(p) as img:
            frame = img.convert("RGB")
            if tile_size > 0:
                frame = frame.resize((tile_size, tile_size), Image.Resampling.LANCZOS)
            images.append(frame)
    return images


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


def _load_label_font(font_path: Optional[Path], size: int) -> ImageFont.ImageFont:
    if font_path is not None:
        return ImageFont.truetype(str(font_path), size)
    try:
        # PIL commonly ships with DejaVu as a scalable fallback.
        return ImageFont.truetype("DejaVuSans.ttf", size)
    except OSError:
        return ImageFont.load_default()


def _build_global_palette_image(frames: Sequence[Image.Image], sample_step: int = 1) -> Image.Image:
    sampled = frames[:: max(sample_step, 1)] or [frames[0]]
    tile_w, tile_h = sampled[0].size
    strip = Image.new("RGB", (tile_w * len(sampled), tile_h), color=(255, 255, 255))
    for i, fr in enumerate(sampled):
        strip.paste(fr, (i * tile_w, 0))
    return strip.quantize(colors=256, method=Image.Quantize.MEDIANCUT, dither=Image.Dither.NONE)


def _quantize_with_palette(
    frames: Sequence[Image.Image],
    palette_image: Image.Image,
    dither_mode: str,
) -> List[Image.Image]:
    dither = Image.Dither.FLOYDSTEINBERG if dither_mode == "floyd" else Image.Dither.NONE
    return [
        fr.quantize(palette=palette_image, dither=dither)
        for fr in frames
    ]


def _save_gif(images: Sequence[Image.Image], output_path: Path, fps: float, loop: int, disposal: int = 2) -> None:
    if not images:
        raise ValueError(f"No images to save for {output_path}")
    duration_ms = int(round(1000.0 / max(fps, 0.1)))
    images[0].save(
        output_path,
        save_all=True,
        append_images=images[1:],
        duration=duration_ms,
        loop=loop,
        optimize=False,
        disposal=disposal,
    )


def _save_webp(
    images: Sequence[Image.Image],
    output_path: Path,
    fps: float,
    loop: int,
    mode: str,
    quality: int,
) -> None:
    if not images:
        raise ValueError(f"No images to save for {output_path}")
    duration_ms = int(round(1000.0 / max(fps, 0.1)))
    images[0].save(
        output_path,
        save_all=True,
        append_images=images[1:],
        duration=duration_ms,
        loop=loop,
        format="WEBP",
        lossless=(mode == "lossless"),
        quality=max(0, min(100, quality)),
        method=6,
    )


def _draw_centered_label(
    draw: ImageDraw.ImageDraw,
    text: str,
    x_center: int,
    y_top: int,
    max_width: int,
    font_path: Optional[Path],
) -> None:
    # Match build_envs_overview_labeled.py defaults (roughly 36 * 0.9).
    font_size = 32
    font = _load_label_font(font_path, font_size)
    bbox = draw.textbbox((0, 0), text, font=font)
    text_w = bbox[2] - bbox[0]
    text_h = bbox[3] - bbox[1]
    while text_w > max_width and font_size > 10:
        font_size -= 1
        font = _load_label_font(font_path, font_size)
        bbox = draw.textbbox((0, 0), text, font=font)
        text_w = bbox[2] - bbox[0]
        text_h = bbox[3] - bbox[1]

    x = x_center - text_w // 2
    y = y_top + text_h
    draw.text((x, y - text_h), text, fill=(20, 20, 20), font=font)


def _build_grid_frames(
    frames_by_slug: Dict[str, List[Image.Image]],
    label_by_slug: Dict[str, str],
    tile_size: int,
    font_path: Optional[Path],
) -> List[Image.Image]:
    cols = 4
    rows = 2
    gap_x = 96
    gap_y = 112
    outer_pad = 56
    label_h = 44

    ordered_slugs = [
        "object_goal",
        "object_interaction",
        "property_comparison",
        "object_comparison",
        "block_lifting",
        "cube_pushing",
        "object_reaching",
        "defend_the_line",
    ]
    lengths = [len(frames_by_slug[s]) for s in ordered_slugs]
    total_frames = max(lengths)
    canvas_w = outer_pad * 2 + cols * tile_size + (cols - 1) * gap_x
    canvas_h = outer_pad * 2 + rows * (label_h + tile_size) + (rows - 1) * gap_y

    grid_frames: List[Image.Image] = []
    for i in range(total_frames):
        canvas = Image.new("RGB", (canvas_w, canvas_h), color=(255, 255, 255))
        draw = ImageDraw.Draw(canvas)
        for idx, slug in enumerate(ordered_slugs):
            r = idx // cols
            c = idx % cols
            row_top = outer_pad + r * (label_h + tile_size + gap_y)
            img_top = row_top + label_h
            x_left = outer_pad + c * (tile_size + gap_x)
            x_center = x_left + tile_size // 2

            env_frames = frames_by_slug[slug]
            # Each environment loops independently inside the grid.
            frame = env_frames[i % len(env_frames)]
            if frame.size != (tile_size, tile_size):
                frame = frame.resize((tile_size, tile_size), Image.Resampling.LANCZOS)
            canvas.paste(frame, (x_left, img_top))
            _draw_centered_label(
                draw=draw,
                text=label_by_slug[slug],
                x_center=x_center,
                y_top=row_top,
                max_width=tile_size - 8,
                font_path=font_path,
            )
        grid_frames.append(canvas)
    return grid_frames


def main() -> None:
    args = parse_args()
    root_dir = args.root_dir.resolve()
    output_dir = args.output_dir.resolve()
    font_path = _find_roboto_font(args.font_path)
    output_dir.mkdir(parents=True, exist_ok=True)
    extension = ".webp" if args.output_format == "webp" else ".gif"

    frames_by_slug: Dict[str, List[Image.Image]] = {}
    labels_by_slug: Dict[str, str] = {}
    skipped: List[str] = []
    for label, slug, visuals_dir_name in _iter_envs():
        frames_dir = root_dir / visuals_dir_name / "video_frames"
        if not frames_dir.exists():
            skipped.append(f"{visuals_dir_name}: missing video_frames")
            continue

        frame_paths = _collect_frames(frames_dir, args.frame_glob, args.max_frames)
        if not frame_paths:
            skipped.append(f"{visuals_dir_name}: no frames matching {args.frame_glob}")
            continue

        frames = _load_frames(frame_paths, args.tile_size)
        frames_by_slug[slug] = frames
        labels_by_slug[slug] = label

    if not frames_by_slug:
        raise RuntimeError("No valid environment frame sets found. Check directories and frame glob.")

    built: List[Path] = []
    if not args.skip_individual:
        for _, slug, _ in _iter_envs():
            if slug not in frames_by_slug:
                continue
            frames = frames_by_slug[slug]
            output_path = output_dir / f"{slug}{extension}"
            if args.output_format == "webp":
                _save_webp(frames, output_path, args.fps, args.loop, args.webp_mode, args.webp_quality)
                frames_count = len(frames)
            else:
                palette = _build_global_palette_image(frames, sample_step=1)
                frames_quant = _quantize_with_palette(frames, palette, args.dither)
                _save_gif(frames_quant, output_path, args.fps, loop=args.loop)
                frames_count = len(frames_quant)
            built.append(output_path)
            print(f"Saved {output_path} ({frames_count} frames)")

    if not args.skip_grid:
        required = [slug for _, slug, _ in _iter_envs()]
        missing_for_grid = [slug for slug in required if slug not in frames_by_slug]
        if missing_for_grid:
            raise RuntimeError(
                "Cannot build grid GIF because some environments are missing: "
                + ", ".join(missing_for_grid)
            )
        grid_frames = _build_grid_frames(frames_by_slug, labels_by_slug, args.grid_tile_size, font_path)
        grid_name = args.grid_output_name
        grid_path = output_dir / (grid_name if Path(grid_name).suffix else f"{grid_name}{extension}")
        if args.output_format == "webp":
            _save_webp(grid_frames, grid_path, args.fps, args.loop, args.webp_mode, args.webp_quality)
            grid_count = len(grid_frames)
        else:
            grid_palette = _build_global_palette_image(grid_frames, sample_step=1)
            grid_quant = _quantize_with_palette(grid_frames, grid_palette, args.dither)
            _save_gif(grid_quant, grid_path, args.fps, loop=args.loop)
            grid_count = len(grid_quant)
        built.append(grid_path)
        print(f"Saved {grid_path} ({grid_count} frames)")

    if not built:
        raise RuntimeError("No GIFs were generated because both modes were skipped.")

    if skipped:
        print("\nSkipped:")
        for item in skipped:
            print(f"- {item}")

    if font_path is not None:
        print(f"\nUsing font: {font_path}")
    else:
        print("\nRoboto font not found, fallback to default PIL font.")


if __name__ == "__main__":
    main()
