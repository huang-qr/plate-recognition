"""
train_crnn.py —— 用 PyTorch 训练 CRNN + CTC 车牌识别模型
用法：python train_crnn.py 训练轮数
"""
import sys

import cv2
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

# OpenCV 自己也会开多线程，和 DataLoader 的多进程一起用时偶尔会互相等待、卡死。
# 关掉 OpenCV 的多线程就能避免（读小图本来也用不上）
cv2.setNumThreads(0)

ROOT = "/kaggle/working/plate/rec"
MODEL_PATH = "/kaggle/working/crnn_best.pt"

# 所有可能出现的字符。编号从 1 开始，0 留给 CTC 的「空白」
CHARS = ("皖沪津渝冀晋蒙辽吉黑苏浙京闽赣鲁豫鄂湘粤桂琼川贵云藏陕甘青宁新警学"
         "ABCDEFGHJKLMNOPQRSTUVWXYZ0123456789")
IMG_WIDTH, IMG_HEIGHT = 160, 48


# ==================================================
# 第一部分：数据
# ==================================================

def preprocess(img):
    """把车牌图缩放到 160x48，转成 3x48x160、数值在 0~1 之间的张量"""
    img = cv2.resize(img, (IMG_WIDTH, IMG_HEIGHT))
    tensor = torch.from_numpy(img).permute(2, 0, 1)   # 高x宽x通道 → 通道x高x宽
    return tensor.float() / 255.0


class PlateDataset(Dataset):
    """读 rec/train.txt 这样的列表，每次返回 (图片张量, 标签编号, 车牌号文字)"""

    def __init__(self, split, augment=False):
        with open(f"{ROOT}/{split}.txt", encoding="utf-8") as f:
            self.items = [line.rstrip("\n").split("\t") for line in f if line.strip()]
        self.augment = augment

    def __len__(self):
        return len(self.items)

    def __getitem__(self, i):
        path, plate = self.items[i]
        img = cv2.imread(f"{ROOT}/{path}")

        if self.augment:   # 数据增强：随机调亮度，让模型适应不同光照
            brightness = np.random.uniform(0.6, 1.4)
            img = np.clip(img * brightness, 0, 255).astype(np.uint8)

        label = torch.tensor([CHARS.index(c) + 1 for c in plate])
        return preprocess(img), label, plate


def collate(batch):
    """把一批样本拼起来。CTC 要求所有标签首尾相连成一长串，再单独给出每个标签的长度"""
    imgs, labels, plates = zip(*batch)
    imgs = torch.stack(imgs)
    label_lengths = torch.tensor([len(label) for label in labels])
    labels = torch.cat(labels)
    return imgs, labels, label_lengths, list(plates)


# ==================================================
# 第二部分：模型
# ==================================================

def conv_block(in_channels, out_channels, pool=None):
    """卷积 + 批归一化 + ReLU，需要的话再加一个池化层缩小尺寸"""
    layers = [
        nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
        nn.BatchNorm2d(out_channels),
        nn.ReLU(inplace=True),
    ]
    if pool:
        layers.append(nn.MaxPool2d(pool))
    return layers


class CRNN(nn.Module):
    def __init__(self, num_classes=len(CHARS) + 1):   # +1 是 CTC 的空白
        super().__init__()
        self.cnn = nn.Sequential(
            *conv_block(3, 64, pool=(2, 2)),       # 48x160 → 24x80
            *conv_block(64, 128, pool=(2, 2)),     # → 12x40
            *conv_block(128, 256),
            *conv_block(256, 256, pool=(2, 1)),    # → 6x40（只压高度，保留宽度方向的 40 列）
            *conv_block(256, 512, pool=(2, 1)),    # → 3x40
        )
        self.rnn = nn.LSTM(512 * 3, 256, num_layers=2, bidirectional=True, batch_first=True)
        self.fc = nn.Linear(256 * 2, num_classes)  # 双向 LSTM 输出是 256x2

    def forward(self, x):
        features = self.cnn(x)                       # B, 512, 3, 40
        b, c, h, w = features.shape
        features = features.permute(0, 3, 1, 2)      # B, 40, 512, 3：把宽度放到第 2 维
        features = features.reshape(b, w, c * h)     # B, 40, 1536：从左到右 40 个时间步
        output, _ = self.rnn(features)               # B, 40, 512
        return self.fc(output)                       # B, 40, 类别数


# ==================================================
# 第三部分：解码和评估
# ==================================================

def decode(logits):
    """CTC 贪心解码：每一列取概率最大的字符，合并相邻重复，去掉空白（编号 0）"""
    plates = []
    for sequence in logits.argmax(dim=2).tolist():
        chars = []
        previous = 0
        for k in sequence:
            if k != 0 and k != previous:
                chars.append(CHARS[k - 1])
            previous = k
        plates.append("".join(chars))
    return plates


@torch.no_grad()
def evaluate(model, loader, device):
    """整串车牌号完全正确才算对"""
    model.eval()
    right = total = 0
    for imgs, _, _, plates in loader:
        preds = decode(model(imgs.to(device)))
        right += sum(pred == plate for pred, plate in zip(preds, plates))
        total += len(plates)
    model.train()
    return right / total


# ==================================================
# 第四部分：训练
# ==================================================

def main():
    epochs = int(sys.argv[1]) if len(sys.argv) > 1 else 30
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # 防卡死的两个设置：
    #   persistent_workers=True  读数据的子进程一直保留，不用每轮重新创建
    #   timeout=300              5 分钟还拿不到一批数据就直接报错，而不是一直卡着
    loader_args = dict(num_workers=2, persistent_workers=True, timeout=300, collate_fn=collate)
    train_loader = DataLoader(PlateDataset("train", augment=True), batch_size=128, shuffle=True, **loader_args)
    val_loader = DataLoader(PlateDataset("val"), batch_size=256, **loader_args)
    test_loader = DataLoader(PlateDataset("test"), batch_size=256, **loader_args)

    model = CRNN().to(device)
    ctc_loss = nn.CTCLoss(blank=0, zero_infinity=True)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)  # 学习率逐渐变小

    best_acc = -1
    for epoch in range(1, epochs + 1):
        for imgs, labels, label_lengths, _ in train_loader:
            logits = model(imgs.to(device))                        # B, T, C
            log_probs = logits.log_softmax(2).permute(1, 0, 2)     # T, B, C（CTCLoss 要求的形状）
            input_lengths = torch.full((imgs.size(0),), log_probs.size(0), dtype=torch.long)
            loss = ctc_loss(log_probs, labels.to(device), input_lengths, label_lengths)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        scheduler.step()

        val_acc = evaluate(model, val_loader, device)
        print(f"epoch {epoch}/{epochs}  loss {loss.item():.4f}  val_acc {val_acc:.2%}", flush=True)
        if val_acc > best_acc:                                     # 只保存验证集上最好的那一轮
            best_acc = val_acc
            torch.save(model.state_dict(), MODEL_PATH)

    model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
    test_acc = evaluate(model, test_loader, device)

    from prepare_data import save_result
    save_result(f"【CRNN 识别】测试集识别准确率（真实框截图）：{test_acc:.2%}")


if __name__ == "__main__":
    main()