# baseline_paddle.py —— 直接用 PaddleOCR 通用识别模型读车牌截图，作为基线
from paddleocr import TextRecognition

ROOT = "/kaggle/working/plate/rec"

with open(f"{ROOT}/test.txt", encoding="utf-8") as f:
    items = [line.rstrip("\n").split("\t") for line in f if line.strip()]

model = TextRecognition()     # 默认 PP-OCRv5 识别模型，第一次运行会自动下载
image_paths = [f"{ROOT}/{path}" for path, _ in items]
results = model.predict(input=image_paths, batch_size=32)   # 按输入顺序返回，每个结果里有 rec_text

right = 0
for (path, true_plate), result in zip(items, results):
    pred = result["rec_text"].replace("·", "").replace(" ", "").upper()   # 去掉车牌中间的点和空格
    right += pred == true_plate

text = f"【PaddleOCR 基线】识别准确率（真实框截图）：{right / len(items):.2%}"
print(text)
with open("/kaggle/working/results.txt", "a", encoding="utf-8") as f:
    f.write(text + "\n")