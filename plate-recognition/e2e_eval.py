"""
e2e_eval.py —— 端到端评测：YOLO 找车牌 → 裁剪 → CRNN 识别
用法：python e2e_eval.py 图片文件夹 场景名
"""
import os
import sys

import cv2
import torch
from ultralytics import YOLO

from prepare_data import parse_name, crop_plate, iou, save_result
from train_crnn import CRNN, preprocess, decode, MODEL_PATH

test_dir = sys.argv[1] if len(sys.argv) > 1 else "/kaggle/working/plate/images/test"
scene = sys.argv[2] if len(sys.argv) > 2 else "正常场景"
device = "cuda" if torch.cuda.is_available() else "cpu"

# 加载两个训练好的模型
detector = YOLO("/kaggle/working/runs/det/weights/best.pt")
recognizer = CRNN().to(device).eval()
recognizer.load_state_dict(torch.load(MODEL_PATH, map_location=device))

wrong_dir = f"/kaggle/working/wrong_cases/{scene}"
os.makedirs(wrong_dir, exist_ok=True)

total = located = recognized = 0
for filename in sorted(os.listdir(test_dir)):
    true_box, true_plate = parse_name(filename)
    img = cv2.imread(os.path.join(test_dir, filename))
    total += 1

    # 1. 检测：取置信度最高的框；一个框都没找到，定位和识别都算错
    boxes = detector(img, verbose=False)[0].boxes
    if len(boxes) == 0:
        continue
    pred_box = boxes.xyxy[boxes.conf.argmax()].tolist()
    if iou(pred_box, true_box) >= 0.7:
        located += 1

    # 2. 裁剪 + 识别
    crop = crop_plate(img, pred_box)
    if crop.size == 0:
        continue
    with torch.no_grad():
        pred_plate = decode(recognizer(preprocess(crop).unsqueeze(0).to(device)))[0]

    if pred_plate == true_plate:
        recognized += 1
    else:   # 保存认错的图，文件名里写上「正确答案_pred_预测结果」
        cv2.imwrite(f"{wrong_dir}/{true_plate}_pred_{pred_plate}.jpg", crop)

save_result(f"【端到端 · {scene}】共 {total} 张 | "
            f"定位准确率（IoU≥0.7）{located / total:.2%} | 端到端识别准确率 {recognized / total:.2%}")