# -*- coding: utf-8 -*-
"""
VGG16 图形分类 · 本地 CPU 版（Windows 友好，num_workers=0 避免多进程报错）
结果会写入 results.json，并保存模型 vgg16_shapes.pth
"""
import os, sys, io, math, time, json, random
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", line_buffering=True)
import numpy as np
from PIL import Image, ImageDraw
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, models

SEED = 42
random.seed(SEED); np.random.seed(SEED); torch.manual_seed(SEED)
torch.set_num_threads(os.cpu_count() or 4)
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
print("device:", DEVICE, "| threads:", torch.get_num_threads(), flush=True)

CLASSES = ["circle", "square", "triangle", "star"]
IMG_SIZE = 64
N_TRAIN_PER_CLASS = 800
N_TEST_PER_CLASS = 200
EPOCHS = 8
BATCH_SIZE = 64
LR = 1e-3


def draw_shape(kind, size=IMG_SIZE, rng=None):
    rng = rng or random
    bg = tuple(rng.randint(200, 255) for _ in range(3))
    img = Image.new("RGB", (size, size), bg)
    d = ImageDraw.Draw(img)
    m = rng.randint(6, 16)
    box = [m, m, size - m, size - m]
    color = tuple(rng.randint(0, 190) for _ in range(3))
    if kind == "circle":
        d.ellipse(box, fill=color)
    elif kind == "square":
        w = rng.randint(m, size - m)
        x0 = rng.randint(2, size - w - 2); y0 = rng.randint(2, size - w - 2)
        d.rectangle([x0, y0, x0 + w, y0 + w], fill=color)
    elif kind == "triangle":
        d.polygon([(size // 2, m), (m, size - m), (size - m, size - m)], fill=color)
    elif kind == "star":
        cx = cy = size // 2
        r = size // 2 - m
        pts = []
        for i in range(10):
            ang = -math.pi / 2 + i * math.pi / 5
            rr = r if i % 2 == 0 else r * 0.45
            pts.append((cx + rr * math.cos(ang), cy + rr * math.sin(ang)))
        d.polygon(pts, fill=color)
    return img


class ShapeDataset(Dataset):
    def __init__(self, n_per_class, seed):
        self.items = []
        rng = random.Random(seed)
        for ci, name in enumerate(CLASSES):
            for _ in range(n_per_class):
                self.items.append((draw_shape(name, rng=rng), ci))
        rng.shuffle(self.items)
        self.tf = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5]),
        ])

    def __len__(self):
        return len(self.items)

    def __getitem__(self, idx):
        img, y = self.items[idx]
        return self.tf(img), y


print("正在生成数据集 ...", flush=True)
train_ds = ShapeDataset(N_TRAIN_PER_CLASS, seed=1)
test_ds = ShapeDataset(N_TEST_PER_CLASS, seed=9999)
train_ld = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True, num_workers=0)
test_ld = DataLoader(test_ds, batch_size=128, shuffle=False, num_workers=0)
print(f"训练集 {len(train_ds)} 张 / 测试集 {len(test_ds)} 张 / {len(CLASSES)} 类", flush=True)

model = models.vgg16(weights=None)
model.classifier[6] = nn.Linear(4096, len(CLASSES))
model = model.to(DEVICE)
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=LR)


def evaluate(m, loader):
    m.eval()
    correct = total = 0
    with torch.no_grad():
        for x, y in loader:
            x, y = x.to(DEVICE), y.to(DEVICE)
            pred = m(x).argmax(1)
            correct += (pred == y).sum().item()
            total += y.size(0)
    return correct / total


best_acc = 0.0
t_start = time.time()
history = []
for ep in range(EPOCHS):
    model.train()
    loss_sum = correct = total = 0
    for x, y in train_ld:
        x, y = x.to(DEVICE), y.to(DEVICE)
        optimizer.zero_grad()
        out = model(x)
        loss = criterion(out, y)
        loss.backward()
        optimizer.step()
        loss_sum += loss.item() * x.size(0)
        correct += (out.argmax(1) == y).sum().item()
        total += x.size(0)
    acc = evaluate(model, test_ld)
    best_acc = max(best_acc, acc)
    history.append({"epoch": ep + 1, "train_loss": round(loss_sum / total, 4),
                    "train_acc": round(correct / total, 4), "test_acc": round(acc, 4)})
    print(f"[epoch {ep + 1}/{EPOCHS}] train_loss={loss_sum / total:.4f} "
          f"train_acc={correct / total:.4f}  test_acc={acc:.4f}  "
          f"({(time.time() - t_start) / 60:.1f} min)", flush=True)

print(f"\n>>> 真实测试集准确率: {best_acc * 100:.2f}%", flush=True)

# -------- 单张推理耗时（CPU，batch=1）--------
model.eval()
dummy = torch.randn(1, 3, IMG_SIZE, IMG_SIZE).to(DEVICE)
with torch.no_grad():
    for _ in range(10):
        model(dummy)
    t0 = time.time()
    for _ in range(50):
        model(dummy)
    ms = (time.time() - t0) / 50 * 1000
print(f">>> 单张推理平均耗时: {ms:.1f} ms （{DEVICE}，batch=1，含前向计算）", flush=True)

torch.save(model.state_dict(), "vgg16_shapes.pth")
print("已保存模型 -> vgg16_shapes.pth", flush=True)

with open("results.json", "w", encoding="utf-8") as f:
    json.dump({
        "classes": CLASSES, "img_size": IMG_SIZE, "epochs": EPOCHS,
        "batch_size": BATCH_SIZE, "lr": LR,
        "n_train": len(train_ds), "n_test": len(test_ds),
        "device": DEVICE, "threads": torch.get_num_threads(),
        "best_test_acc": round(best_acc * 100, 2),
        "single_infer_ms": round(ms, 1),
        "history": history,
    }, f, ensure_ascii=False, indent=2)
print("已保存 results.json", flush=True)
