# 基于 VGG16 的图形分类系统

> 深度学习课程项目 · 图像分类 · PyTorch 实现

使用 VGG16 卷积神经网络对 **4 类几何图形**（圆形 / 方形 / 三角形 / 星形）进行分类，
并完成「数据生成 → 模型训练 → 精度评估 → 鲁棒性测试 → 推理性能测试」的完整闭环。

---

## 一、实验结果（实测）

| 指标 | 数值 | 说明 |
|---|---|---|
| 测试集准确率 | **100.00%** | 800 张干净测试样本（4 类 × 200 张） |
| 抗干扰准确率 | **59.62%** | 叠加 旋转 ±40° / 高斯噪声(σ=25) / 随机遮挡 / 亮度扰动 |
| 单张推理耗时 | **40.6 ms** | CPU（16 线程），batch=1，纯模型前向 |
| 训练配置 | VGG16 从头训练，Adam，lr 1e-3 → 3e-4 余弦退火 | 约 36 分钟（CPU） |

### 混淆矩阵（干净测试集，800 张全部分类正确）

```
              circle    square  triangle      star
circle           200         0         0         0
square             0       200         0         0
triangle           0         0       200         0
star               0         0         0       200
```

### 鲁棒性测试（叠加扰动后）

```
              circle    square  triangle      star
circle           161        39         0         0
square            35       147         9         9
triangle          32        90        78         0
star              25        81         3        91
```

**结论**：模型在干净数据上表现优异，但对**旋转、遮挡等几何变换敏感**
（三角形 / 星形在旋转后易被误判为圆形、方形），说明训练时数据增强不足 ——
后续可加入更强的几何增强与对抗训练提升鲁棒性。

---

## 二、网络结构

- 主干：**VGG16**（13 个卷积层 + 3 个全连接层）
- 输入：64 × 64 × 3
- 输出：4 类
- 损失函数：交叉熵
- 优化器：Adam

```
Input(64×64×3)
  └─ VGG16 features (conv ×13 + maxpool ×5)
      └─ FC 4096 → ReLU → Dropout
          └─ FC 4096 → ReLU → Dropout
              └─ FC 4  (分类输出)
```

---

## 三、运行方式

```bash
pip install torch torchvision pillow numpy

# 直接运行（CPU 约 20~40 分钟；GPU 数分钟）
python vgg16_shape_classifier.py
```

或直接打开 `VGG16_图形分类.ipynb` 逐个代码格执行。

> 数据集由程序 **自动生成**（带随机位置 / 大小 / 颜色 / 背景的合成图形），
> 无需下载任何外部数据，秒级生成 3200 张训练图 + 800 张测试图。

---

## 四、目录结构

```
.
├── README.md                  # 本文件
├── VGG16_图形分类.ipynb        # 完整实验 Notebook
├── vgg16_shape_classifier.py  # 等价的可执行脚本
├── requirements.txt           # 依赖
├── results.json               # 训练过程记录（每个 epoch 的 loss / acc）
├── final_metrics.json         # 干净测试集最终指标 + 混淆矩阵
├── hard_metrics.json          # 鲁棒性测试指标 + 混淆矩阵
└── .gitignore                 # 排除权重、数据集等大文件
```

> 训练好的权重 `vgg16_shapes.pth`（约 512 MB）超出 GitHub 单文件 100 MB 限制，
> 未上传；如需复现请按上面步骤自行训练（CPU 约半小时）。

---

## 五、后续优化方向

- [ ] 换用真实数据集（CIFAR-10 / TrashNet）验证泛化能力
- [ ] 使用迁移学习（ImageNet 预训练权重）加速收敛
- [ ] 增加旋转 / 遮挡 / 噪声等数据增强，提升鲁棒性
- [ ] 模型量化与 ONNX 导出，部署到 Web 端推理服务
