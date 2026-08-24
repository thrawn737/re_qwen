# merge_windows.py
import os
import sys
import glob
import re
import argparse

def merge_window_files(folder_path, output_file):
    """
    合并指定文件夹下所有 L01_window_*.txt 文件为一个完整报告
    """
    pattern = os.path.join(folder_path, "L01_window_*.txt")
    files = glob.glob(pattern)
    
    if not files:
        print(f"错误: {folder_path} 下没有找到 L01_window_*.txt 文件")
        return False
    
    def get_window_num(filepath):
        basename = os.path.basename(filepath)
        match = re.search(r'window_(\d+)', basename)
        return int(match.group(1)) if match else 0
    
    files.sort(key=get_window_num)
    
    with open(output_file, "w", encoding="utf-8") as out_f:
        out_f.write(f"手术ID: L01\n")
        out_f.write(f"总窗口数: {len(files)}\n")
        out_f.write(f"{'='*60}\n\n")
        
        for filepath in files:
            with open(filepath, "r", encoding="utf-8") as in_f:
                content = in_f.read().strip()
                basename = os.path.basename(filepath)
                out_f.write(f"[{basename}]\n")
                out_f.write(content)
                out_f.write("\n\n" + "="*60 + "\n\n")
    
    print(f"合并完成: {output_file} (共 {len(files)} 个窗口)")
    return True

def main():
    parser = argparse.ArgumentParser(description="合并窗口文件")
    parser.add_argument("--folder", type=str, required=True, help="包含 L01_window_*.txt 的文件夹路径")
    parser.add_argument("--output", type=str, required=True, help="输出文件路径")
    args = parser.parse_args()
    
    if not os.path.exists(args.folder):
        print(f"错误: 文件夹不存在: {args.folder}")
        sys.exit(1)
    
    os.makedirs(os.path.dirname(os.path.abspath(args.output)), exist_ok=True)
    merge_window_files(args.folder, args.output)

if __name__ == "__main__":
    main()


#     # 合并带 ReVisiT 的
# python merge_windows.py --folder ./output/surgery_captions --output ./output/surgery_captions/L01_merged_summary.txt

# # 合并不带 ReVisiT 的
# python merge_windows.py --folder ./output/surgery_captions_noR --output ./output/surgery_captions_noR/L01_merged_summary.txt