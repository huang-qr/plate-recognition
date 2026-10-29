"""
cv_baseline.py —— 传统方法定位车牌：蓝色区域 + 形态学闭运算 + 轮廓筛选
用法：python cv_baseline.py 图片文件夹
"""
import os
import sys

import cv2

from prepare_data import parse_name, iou, save_result


def locate(img):
    """返回找到的车牌框 (left, top, right, bottom)，找不到返回 None"""
    # 1. 找蓝色区域（CCPD 绝大多数是蓝牌）
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    mask = cv2.inRange(hsv, (100, 80, 60), (130, 255, 255))

    # 2. 闭运算：把字符之间的缝隙连起来，让整块车牌变成一个整体
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (17, 5))
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)

    # 3. 找轮廓，按车牌的长宽比和最小面积筛选，留下面积最大的
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    best = None
    for contour in contours:
        x, y, w, h = cv2.boundingRect(contour)
        looks_like_plate = 2 < w / h < 5.5 and w * h > 1500
        if looks_like_plate and (best is None or w * h > best[2] * best[3]):
            best = (x, y, w, h)

    if best is None:
        return None
    x, y, w, h = best
    return (x, y, x + w, y + h)


if __name__ == "__main__":
    test_dir = sys.argv[1] if len(sys.argv) > 1 else "/kaggle/working/plate/images/test"
    filenames = sorted(os.listdir(test_dir))

    located = 0
    for filename in filenames:
        pred_box = locate(cv2.imread(os.path.join(test_dir, filename)))
        true_box, _ = parse_name(filename)
        if pred_box is not None and iou(pred_box, true_box) >= 0.7:
            located += 1

    save_result(f"【OpenCV 传统方法 · 正常场景】共 {len(filenames)} 张 | "
                f"定位准确率（IoU≥0.7）{located / len(filenames):.2%}")