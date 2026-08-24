from huggingface_hub import snapshot_download
import argparse
import os

revisit_path = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
def parse_args():
    parser = argparse.ArgumentParser(description="LLaVA or QwenVL")
    parser.add_argument("--download_path", type=str, default=os.path.join(revisit_path, "data"), help="download path")
    parser.add_argument("--llava" , action='store_true', help="download llava-v1.5-7b")
    parser.add_argument("--qwenvl" , action='store_true', help="download Qwen2.5-VL-7B-Instruct")
    parser.add_argument("--internvl" , action='store_true', help="download InternVL3-8B")

    args = parser.parse_known_args()[0]
    return args

def main():
    args = parse_args()
    os.makedirs(args.download_path, exist_ok=True)

    if args.llava:
        snapshot_download(repo_id="liuhaotian/llava-v1.5-7b", local_dir=os.path.join(args.download_path, "llava-v1.5-7b"))
    if args.qwenvl:
        snapshot_download(repo_id="Qwen/Qwen2.5-VL-7B-Instruct", local_dir=os.path.join(args.download_path, "Qwen2.5-VL-7B-Instruct"))
    if args.internvl:
        snapshot_download(repo_id="OpenGVLab/InternVL3-8B", local_dir=os.path.join(args.download_path, "InternVL3-8B"))

if __name__ == "__main__":
    main()