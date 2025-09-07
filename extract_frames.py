import argparse
import concurrent.futures as cf
from pathlib import Path

import cv2
from tqdm import tqdm


def sample_video(video_path: Path, extract_dir: Path, sampling_period: int = 6, jobs: int = 1):
    vid = cv2.VideoCapture(str(video_path))
    extract_dir.mkdir(exist_ok=True)
    n_frames = int(vid.get(cv2.CAP_PROP_FRAME_COUNT))
    parallel_saver = cf.ThreadPoolExecutor(max_workers=jobs)

    for frame_idx in tqdm(
        range(n_frames),
        desc="sampling video",
        leave=False,
        bar_format="{l_bar}{bar:10}{r_bar}{bar:-10b}",
    ):
        _, frame = vid.read()
        # print(frame_idx // sampling_period, frame_idx % sampling_period)

        # If the segmentation does not exist for a frame, there is no need to generate it
        segment_path = video_path.parent / f"segmentation/{frame_idx:09d}.png"
        if not segment_path.exists():
            # print("Segmentation does not exist. Skipping")
            pass
        else:
            if frame_idx % sampling_period == 0:
                parallel_saver.submit(
                    cv2.imwrite,
                    str(extract_dir / f"{frame_idx:09d}.png"),
                    frame,
                )

    vid.release()
    parallel_saver.shutdown(wait=True)


def main(args):
    video_fps = 60  # All sarrarp50 videos are recorded at 60 fps
    sampling_period: int = video_fps // args.frequency

    # Find all files that need to be processed
    if not args.recursive:
        video_dirs = [Path(args.data_dir).resolve()]
    else:
        video_dirs = [v_p.parent for v_p in Path(args.data_dir).rglob("*video_left.avi")]

    # Validate paths
    for directory in video_dirs:
        if not (directory.exists() and (directory / "video_left.avi").exists()):
            print(
                f"{directory} is not a video directory. Please make sure the video directory path is correct"
            )

    for directory in tqdm(
        video_dirs,
        desc="unpacking dataset",
        bar_format="{l_bar}{bar:10}{r_bar}{bar:-10b}",
    ):
        rgb_dir = directory / "rgb"
        sample_video(directory / "video_left.avi", rgb_dir, sampling_period, args.jobs)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("data_dir", help="path to the video directory")
    parser.add_argument(
        "-f",
        "--frequency",
        type=int,
        help="sampling rate in Hz",
        choices=[1, 10],
        default=10,
    )
    parser.add_argument(
        "-r",
        "--recursive",
        help="search recursively for video directories that have video_left.avi as a child",
        action="store_true",
    )
    parser.add_argument(
        "-j",
        "--jobs",
        help="number of parallel workers to use when saving images",
        default=4,
        type=int,
    )

    SystemExit(main(parser.parse_args()))
