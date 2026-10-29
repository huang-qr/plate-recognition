"""
make_hard.py —— 从三个困难子集各随机抽 500 张，复制到 plate/hard/
"""
import os
import random
import shutil

from prepare_data import CCPD_DIR, OUTPUT_DIR, run_in_threads

random.seed(0)
for subset in ["tilt", "weather", "challenge"]:
    src_dir = f"{CCPD_DIR}/ccpd_{subset}"
    dst_dir = f"{OUTPUT_DIR}/hard/{subset}"
    if not os.path.isdir(src_dir):
        print(f"没有找到 {src_dir}，跳过")
        continue

    os.makedirs(dst_dir, exist_ok=True)
    all_files = sorted(f for f in os.listdir(src_dir) if f.endswith(".jpg"))
    chosen = random.sample(all_files, 500)
    jobs = [(f"{src_dir}/{f}", dst_dir) for f in chosen]
    run_in_threads(shutil.copy, jobs, f"[困难场景] {subset}")