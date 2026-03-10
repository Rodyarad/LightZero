import os
import cv2

def extract_frames_from_video(
    video_path: str,
    output_dir: str,
    prefix: str = "frame",
    ext: str = ".jpg",
) -> None:
    os.makedirs(output_dir, exist_ok=True)

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open video: {video_path}")

    frame_idx = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        filename = f"{prefix}_{frame_idx:04d}{ext}"
        out_path = os.path.join(output_dir, filename)
        ok = cv2.imwrite(out_path, frame)
        if not ok:
            print(f"Warning: failed to write {out_path}")
        frame_idx += 1

    cap.release()
    print(f"Saved {frame_idx} frames to {output_dir}")

if __name__ == "__main__":
    video_path = "/home/rodya-rad/Desktop/work/LightZero/visuals/TargetEnv-v0-video-20260306155304-episode-0.mp4"
    output_dir = "visuals/video_frames"

    extract_frames_from_video(video_path, output_dir)