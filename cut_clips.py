import os
import re
import argparse
import subprocess


def parse_groups_txt(groups_txt):
    groups = []
    with open(groups_txt, 'r') as f:
        lines = f.readlines()
    current = {}
    for line in lines:
        line = line.strip()
        if not line:
            continue
        if line.startswith("Group"):
            match = re.search(r'Group\s+(\d+)', line)
            if match:
                gid = int(match.group(1))
                if current:
                    groups.append(current)
                current = {'gid': gid}
        elif line.startswith("start_frame:"):
            current['start_frame'] = int(line.split()[1])
        elif line.startswith("end_frame:"):
            current['end_frame'] = int(line.split()[1])
        elif line.startswith("output_dir:"):
            current['output_dir'] = line.split(' ', 1)[1].strip()
        elif line.startswith("n_sampled:"):
            current['n_sampled'] = int(line.split()[1])
    if current:
        groups.append(current)
    return groups


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--groups_txt", default="./output/groups.txt", help="Grouping file")
    parser.add_argument("--video_path", required=True, help="Original video path")
    parser.add_argument("--output_dir", default="./output/clips", help="Output directory")
    parser.add_argument("--fps", type=float, default=30.0, help="Video FPS")
    args = parser.parse_args()

    groups = parse_groups_txt(args.groups_txt)
    print(f"Read {len(groups)} groups")

    os.makedirs(args.output_dir, exist_ok=True)

    for g in groups:
        start = g['start_frame']
        end = g['end_frame']
        start_sec = start / args.fps
        end_sec = end / args.fps
        duration = end_sec - start_sec
        if duration < 0.1:
            duration = 0.1

        out_file = os.path.join(args.output_dir, f"clip_{g['gid']:03d}.mp4")

        cmd = [
            "ffmpeg",
            "-i", args.video_path,
            "-ss", str(start_sec),
            "-t", str(duration),
            "-c:v", "libx264",
            "-crf", "18",
            "-preset", "fast",
            "-c:a", "aac",
            "-y",
            out_file
        ]
        print(f"  Cutting clip_{g['gid']:03d}: {start} → {end} ({duration:.2f}s)")
        subprocess.run(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)

    print(f"Done, output directory: {args.output_dir}")


if __name__ == "__main__":
    main()