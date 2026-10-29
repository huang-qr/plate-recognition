"""
prepare_data.py —— 把 CCPD 数据集整理成两份训练数据

  1. 检测数据（给 YOLO 用）：plate/images/ 放原图，plate/labels/ 放同名标签 txt
  2. 识别数据（给 CRNN 用）：plate/rec/ 放裁好的车牌小图，rec/train.txt 等记录「图片路径<Tab>车牌号」

这个文件里的 parse_name、crop_plate、iou、save_result 后面几个脚本也会导入使用。
"""
import os
import re
import random
import shutil
from concurrent.futures import ThreadPoolExecutor

import cv2


# ==================================================
# 第一部分：设置（想改路径或数量，只改这里）
# ==================================================

CCPD_DIR = "/kaggle/input/datasets/binh234/ccpd-preprocess/CCPD2019"
SOURCE_DIR = f"{CCPD_DIR}/ccpd_base"
OUTPUT_DIR = "/kaggle/working/plate"
RESULT_FILE = "/kaggle/working/results.txt"

NUM_TRAIN = 8000         # 检测训练集
NUM_VAL = 1000           # 检测验证集
NUM_TEST = 1000          # 检测测试集
NUM_EXTRA_REC = 40000    # 额外给识别模型的图片（只裁车牌，不占多少空间）
NUM_THREADS = 16         # 同时读多少张图


# ==================================================
# 第二部分：车牌字符表
# 文件名里的车牌号是数字编号，要用这三张表翻译成汉字和字母
# ==================================================

# 第 1 位：省份
PROVINCES = ["皖", "沪", "津", "渝", "冀", "晋", "蒙", "辽", "吉", "黑", "苏", "浙",
             "京", "闽", "赣", "鲁", "豫", "鄂", "湘", "粤", "桂", "琼", "川", "贵",
             "云", "藏", "陕", "甘", "青", "宁", "新", "警", "学", "O"]

# 第 2 位：字母（车牌里没有 I 和 O，最后的 "O" 是占位符）
LETTERS = ["A", "B", "C", "D", "E", "F", "G", "H", "J", "K", "L", "M",
           "N", "P", "Q", "R", "S", "T", "U", "V", "W", "X", "Y", "Z", "O"]

# 第 3~7 位：字母或数字
LETTERS_AND_DIGITS = LETTERS[:-1] + ["0", "1", "2", "3", "4", "5", "6", "7", "8", "9", "O"]


# ==================================================
# 第三部分：工具函数
# ==================================================

def parse_name(filename):
    """
    从 CCPD 文件名中读出车牌框和车牌号。

    用 "-" 切开后：
        第 3 段 "352,516_448,547"      → 车牌框：左上角 (352, 516)，右下角 (448, 547)
        第 5 段 "0_0_22_10_26_29_24"   → 车牌号的 7 个字符编号

    返回：((left, top, right, bottom), "皖AYL250")
    """
    name = os.path.basename(filename)   # 去掉前面的文件夹路径
    name = name[:-4]                    # 去掉结尾的 ".jpg"
    parts = name.split("-")

    # --- 车牌框 ---
    # 不同版本的 CCPD 用不同符号分隔坐标（& _ ,），三种都当作分隔符
    box_numbers = re.split(r"[&_,]", parts[2])
    left, top, right, bottom = [int(n) for n in box_numbers]
    box = (left, top, right, bottom)

    # --- 车牌号 ---
    char_ids = [int(n) for n in parts[4].split("_")]
    province = PROVINCES[char_ids[0]]
    letter = LETTERS[char_ids[1]]
    rest = ""
    for char_id in char_ids[2:]:
        rest += LETTERS_AND_DIGITS[char_id]
    plate = province + letter + rest

    return box, plate


def box_to_yolo_label(box, img_width, img_height):
    """
    把车牌框换成 YOLO 标签格式：  类别  中心x  中心y  宽  高
    后四个数都要除以图片宽高，变成 0~1 之间的比例。只有车牌一类，类别固定是 0。
    """
    left, top, right, bottom = box

    center_x = (left + right) / 2 / img_width
    center_y = (top + bottom) / 2 / img_height
    box_width = (right - left) / img_width
    box_height = (bottom - top) / img_height

    return f"0 {center_x:.6f} {center_y:.6f} {box_width:.6f} {box_height:.6f}"


def crop_plate(img, box, pad=0.08):
    """
    按车牌框把车牌裁出来，四周各多留 8% 的边。
    训练和评测都用这个函数，保证裁法一致。
    """
    left, top, right, bottom = box
    plate_width = right - left
    plate_height = bottom - top
    img_height, img_width = img.shape[:2]

    # 往外扩一点，同时不能超出图片边界
    left = max(0, int(left - pad * plate_width))
    top = max(0, int(top - pad * plate_height))
    right = min(img_width, int(right + pad * plate_width))
    bottom = min(img_height, int(bottom + pad * plate_height))

    return img[top:bottom, left:right]


def iou(box_a, box_b):
    """
    交并比 IoU = 两个框重叠的面积 / 两个框合起来的面积。
    1 表示完全重合，0 表示完全不挨着。用来判断定位准不准。
    """
    overlap_width = max(0, min(box_a[2], box_b[2]) - max(box_a[0], box_b[0]))
    overlap_height = max(0, min(box_a[3], box_b[3]) - max(box_a[1], box_b[1]))
    overlap_area = overlap_width * overlap_height

    area_a = (box_a[2] - box_a[0]) * (box_a[3] - box_a[1])
    area_b = (box_b[2] - box_b[0]) * (box_b[3] - box_b[1])
    union_area = area_a + area_b - overlap_area

    return overlap_area / union_area


def run_in_threads(func, jobs, title):
    """
    用多线程对每个 job 调用 func(*job)，按原来的顺序返回结果，并打印进度。
    图片在网络盘上，大部分时间都在等文件，多个线程一起等能快很多。
    """
    results = []
    with ThreadPoolExecutor(max_workers=NUM_THREADS) as pool:
        for n, result in enumerate(pool.map(lambda job: func(*job), jobs), start=1):
            results.append(result)
            if n % 5000 == 0 or n == len(jobs):
                print(f"  {title}: {n}/{len(jobs)}", flush=True)
    return results


def save_result(text):
    """打印一行结果，同时追加到 results.txt，最后统一查看"""
    print(text)
    with open(RESULT_FILE, "a", encoding="utf-8") as f:
        f.write(text + "\n")


# ==================================================
# 第四部分：生成数据
# ==================================================

def split_files():
    """读出所有图片名，打乱后切成 训练 / 验证 / 测试 / 额外识别 四份，互不重叠"""
    all_files = sorted(f for f in os.listdir(SOURCE_DIR) if f.endswith(".jpg"))
    random.seed(0)                      # 固定随机种子，每次切出来的都一样
    random.shuffle(all_files)
    print(f"ccpd_base 共 {len(all_files)} 张")

    end_train = NUM_TRAIN
    end_val = end_train + NUM_VAL
    end_test = end_val + NUM_TEST
    end_extra = end_test + NUM_EXTRA_REC

    return {
        "train": all_files[:end_train],
        "val": all_files[end_train:end_val],
        "test": all_files[end_val:end_test],
        "extra": all_files[end_test:end_extra],
    }


def make_detection_sample(split, filename):
    """处理一张检测图：复制原图 + 写 YOLO 标签"""
    image_path = f"{OUTPUT_DIR}/images/{split}/{filename}"
    shutil.copy(f"{SOURCE_DIR}/{filename}", image_path)

    img = cv2.imread(image_path)        # 读刚复制到本地的那份，比读网络盘快
    img_height, img_width = img.shape[:2]
    box, _ = parse_name(filename)       # 检测只需要框，车牌号用 _ 忽略
    label = box_to_yolo_label(box, img_width, img_height)

    label_path = f"{OUTPUT_DIR}/labels/{split}/{filename.replace('.jpg', '.txt')}"
    with open(label_path, "w") as f:
        f.write(label + "\n")


def make_recognition_sample(split, index, filename):
    """处理一张识别图：裁出车牌小图保存，返回「图片路径<Tab>车牌号」"""
    box, plate = parse_name(filename)
    img = cv2.imread(f"{SOURCE_DIR}/{filename}")
    crop = crop_plate(img, box)

    crop_name = f"{split}/{index:06d}.jpg"     # train/000000.jpg, train/000001.jpg, ...
    cv2.imwrite(f"{OUTPUT_DIR}/rec/{crop_name}", crop)
    return f"{crop_name}\t{plate}"


def main():
    files = split_files()

    # 1. 检测数据
    for split in ["train", "val", "test"]:
        os.makedirs(f"{OUTPUT_DIR}/images/{split}", exist_ok=True)
        os.makedirs(f"{OUTPUT_DIR}/labels/{split}", exist_ok=True)
        jobs = [(split, filename) for filename in files[split]]
        run_in_threads(make_detection_sample, jobs, f"[检测] {split}")

    # 2. 识别数据（训练集多加 4 万张，识别模型需要更多样本）
    rec_files = {
        "train": files["train"] + files["extra"],
        "val": files["val"],
        "test": files["test"],
    }
    for split, filenames in rec_files.items():
        os.makedirs(f"{OUTPUT_DIR}/rec/{split}", exist_ok=True)
        jobs = [(split, i, filename) for i, filename in enumerate(filenames)]
        lines = run_in_threads(make_recognition_sample, jobs, f"[识别] {split}")
        with open(f"{OUTPUT_DIR}/rec/{split}.txt", "w", encoding="utf-8") as f:
            f.write("\n".join(lines) + "\n")

    print("全部完成！")


# 直接运行这个文件时才执行 main()；被别的脚本 import 时不执行
if __name__ == "__main__":
    main()