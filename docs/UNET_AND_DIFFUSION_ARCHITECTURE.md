# 3D U-Net 与 U-Net + Diffusion 模型结构

本文档统一说明本项目当前实现的两种 RNV 分割网络：

- baseline：`UNet3D`；
- diffusion experiment：`DiffusionUNet3D`，即 coarse 3D U-Net 加条件去噪 3D U-Net。

文档以当前代码为准。核心实现位于：

- `src/rnv_seg/models/unet3d.py`
- `src/rnv_seg/models/diffusion_unet3d.py`
- `src/rnv_seg/losses/segmentation.py`
- `configs/base.yaml`
- `configs/experiments/diffusion_unet_oct_octa.yaml`

## 1. 共享输入与输出约定

网络输入统一为五维 tensor：

```text
[batch, channel, depth, width, B-scan]
```

当前训练 patch 为 `[480,224,24]`，因此输入 shape 为：

```text
[N,C,480,224,24]
```

输入通道数由 `data.modalities` 自动决定：

| 实验 | modalities | C |
|---|---|---:|
| OCT | `[oct]` | 1 |
| OCT + OCTA | `[oct,octa]` | 2 |
| Multimodal | `[oct,octa,large_vessel]` | 3 |

单任务模型输出一个 RNV logit channel：

```text
logits: [N,1,D,W,B]
probability = sigmoid(logits)
```

模型内部不执行固定 threshold。训练损失直接接收 logits；验证和推理阶段再执行
sigmoid 和配置中的 threshold。

## 2. Baseline：3D U-Net

### 2.1 总体结构

当前默认参数为：

```yaml
model:
  name: unet3d
  base_channels: 24
  levels: 5
  dropout: 0.1
  out_channels: 1
```

当前恢复表现更好的 24 base channels，并使用 `Dropout3d(0.1)` 与 AdamW
`weight_decay=1e-5`。模型参数保持 float32，训练开启 CUDA AMP；训练 B-scan extent 从
32 降至 24 帧，以较小 activation memory 保留五层 receptive field。16-channel 或旧
32-frame run 必须配合各自的 `resolved_config.yaml` 使用；当前配置需要开启新 run。

网络由五层 encoder、四个 2× downsampling、四个 decoder stage 和一个 1×1×1
输出头组成：

```text
Input [N,C,D,W,B]
  │
  ├─ Encoder 1: ResidualBlock C→24 ───────────────── skip 1
  │    MaxPool3d(2)
  ├─ Encoder 2: ResidualBlock 24→48 ──────────────── skip 2
  │    MaxPool3d(2)
  ├─ Encoder 3: ResidualBlock 48→96 ──────────────── skip 3
  │    MaxPool3d(2)
  ├─ Encoder 4: ResidualBlock 96→192 ─────────────── skip 4
  │    MaxPool3d(2)
  └─ Bottleneck: ResidualBlock 192→384
       │
       ├─ TransposedConv 384→192 + skip 4 → ResidualBlock 384→192
       ├─ TransposedConv 192→96  + skip 3 → ResidualBlock 192→96
       ├─ TransposedConv 96→48   + skip 2 → ResidualBlock 96→48
       ├─ TransposedConv 48→24   + skip 1 → ResidualBlock 48→24
       └─ Conv3d 1×1×1, 24→out_channels
             │
             └─ logits [N,out_channels,D,W,B]
```

每次 encoder pooling 和 decoder up-convolution 都同时作用于 depth、width 和
B-scan 三个方向，因此模型学习的是连续 B-scan 间的三维结构，而不是逐张进行二维分割。

### 2.2 3D ResidualBlock

每个 `ResidualConvBlock(in_channels, out_channels)` 包含主分支和 shortcut：

```text
Main branch:
  Conv3d 3×3×3 → GroupNorm → SiLU → Dropout3d
  → Conv3d 3×3×3 → GroupNorm

Shortcut:
  Identity                                  if Cin = Cout
  Conv3d 1×1×1 → GroupNorm                 if Cin ≠ Cout

Output:
  SiLU(main branch + shortcut)
```

3×3×3 convolution 使用 padding 1，不改变空间尺寸。Dropout 只放在 block 的两次
convolution 之间。1×1×1 projection 使不同 channel 数的输入能够与主分支相加。
Residual shortcut 为梯度提供直接路径，降低五层网络加深后的优化难度。GroupNorm 不依赖
较大的 batch，适合当前 batch size 2 的训练设置。

`ConvBlock` 仍保留给 Foundation/OCTCube 等需要兼容旧 checkpoint 或既有 adapter 的
模型；本文的 baseline、multitask 和 diffusion U-Net 使用 `ResidualConvBlock`。

### 2.3 当前训练 patch 的 feature shape

以 `[N,C,480,224,24]` 为例：

| Stage | Channels | Spatial shape `[D,W,B]` |
|---|---:|---|
| Input | C | `[480,224,24]` |
| Encoder 1 / skip 1 | 24 | `[480,224,24]` |
| Pool 1 | 24 | `[240,112,12]` |
| Encoder 2 / skip 2 | 48 | `[240,112,12]` |
| Pool 2 | 48 | `[120,56,6]` |
| Encoder 3 / skip 3 | 96 | `[120,56,6]` |
| Pool 3 | 96 | `[60,28,3]` |
| Encoder 4 / skip 4 | 192 | `[60,28,3]` |
| Pool 4 | 192 | `[30,14,1]` |
| Bottleneck | 384 | `[30,14,1]` |
| Decoder 1 | 192 | `[60,28,3]` |
| Decoder 2 | 96 | `[120,56,6]` |
| Decoder 3 | 48 | `[240,112,12]` |
| Decoder 4 | 24 | `[480,224,24]` |
| Output logits | 1 | `[480,224,24]` |

如果某个输入维度因 pooling 取整而与 skip tensor 不一致，decoder 会使用 trilinear
interpolation 将 upsampled feature 对齐到 skip 的空间尺寸，再进行 channel 拼接。

### 2.4 Skip connection

Encoder feature 不做相加，而是在 channel 维与 decoder feature 拼接：

```text
decoder_input = concat(upsampled_feature, encoder_skip, dim=channel)
```

这样 decoder 同时获得：

- bottleneck 的高层语义信息；
- encoder 的局部边缘、形状和微小 RNV 细节。

### 2.5 单输出与双输出 U-Net

普通 `UNet3D` 使用一个 `Conv3d(24,1,kernel_size=1)` 输出 RNV logits。

`MultiTaskUNet3D` 共享完全相同的 encoder 和 decoder，但使用两个独立的 1×1×1 head：

```text
shared decoder feature [N,24,D,W,B]
  ├─ rnv head          → [N,1,D,W,B]
  └─ vasculature head  → [N,1,D,W,B]
```

两个输出在 channel 维拼接为 `[rnv,vasculature]`。该双输出结构属于 U-Net 变体；当前
`DiffusionUNet3D` 只支持一个二值输出，不支持该双头模式。

## 3. U-Net + Diffusion

### 3.1 总体结构

`DiffusionUNet3D` 包含两个独立的 3D U-Net：

1. `coarse`：与 baseline 相同的完整 U-Net，生成初始 RNV logits；
2. `refiner`：较小的条件 U-Net，预测加入 mask 的 Gaussian noise。

对于当前 OCT + OCTA 实验：

```text
Image x: [N,2,D,W,B]
  │
  ├─ Coarse UNet3D
  │    input channels = 2
  │    channels = [24,48,96,192,384]
  │    output = coarse_logits [N,1,D,W,B]
  │
  └─ Refiner UNet3D
       input = concat(image, mask_condition, time_channel)
       input channels = 2 + 1 + 1 = 4
       channels = [12,24,48,96,192]
       output = predicted_noise [N,1,D,W,B]
```

Refiner 的基础 channel 数为：

```text
max(8, base_channels / 2)
```

默认 `base_channels=24`，所以 refiner 从 12 channels 开始，计算量小于 coarse U-Net。

### 3.2 Diffusion schedule

当前配置为：

```yaml
model:
  name: diffusion_unet3d
  diffusion_steps: 50
  diffusion_loss_weight: 0.2
```

代码线性生成：

```text
βₜ ∈ [1×10⁻⁴, 0.02],  t = 0,...,49
αₜ = 1 - βₜ
ᾱₜ = product(α₀,...,αₜ)
```

`alpha_bars` 注册为 model buffer，会随 checkpoint 保存并自动移动到 GPU，但不是可训练
参数。

### 3.3 训练阶段

训练时模型同时接收 image `x` 和 ground-truth mask `y`：

```text
                         ┌──────────────────────────────┐
image x ────────────────→│ coarse U-Net → coarse logits │
                         └──────────────────────────────┘

target y + random noise ε + random timestep t
  │
  ├─ yₜ = sqrt(ᾱₜ)y + sqrt(1-ᾱₜ)ε
  └─ concat(x, yₜ, normalized t)
         │
         └─ refiner U-Net → predicted noise ε̂
                 │
                 └─ reconstruct mask estimate ŷ₀
                         │
                         └─ refined logits

final logits = 0.5 × (coarse logits + refined logits)
```

具体步骤：

1. 每个 batch sample 独立随机选择 `t ∈ [0,49]`；
2. 生成与 target 同 shape 的标准 Gaussian noise `ε`；
3. 按 forward diffusion 公式得到 noisy target `yₜ`；
4. 将 `t/(diffusion_steps-1)` 广播为空间 time channel；
5. Refiner 根据 `[image,yₜ,time]` 预测 noise；
6. 根据预测 noise 反推 clean mask estimate；
7. 将其截断到 `[1e-4,1-1e-4]` 后转换为 logits；
8. coarse logits 与 refined logits 等权平均。

训练总损失为：

```text
Ltotal = Lseg(final_logits, target) + 0.2 × MSE(predicted_noise, true_noise)
```

其中当前共享 segmentation loss 为：

```text
Lseg = Dice loss + BCEWithLogits loss + Focal Tversky loss
```

Focal Tversky 当前使用 `alpha=0.6`、`beta=0.4`、`gamma=0.75`；代码中 alpha 对
false positive 加权，因此该设置更强调抑制正常黄斑区域的假阳性。

### 3.4 验证与推理阶段

`model.eval()` 时不会读取 ground-truth target，也不会随机选择 timestep。实际路径为：

```text
image x
  └─ coarse U-Net → coarse logits → sigmoid → coarse probability
                                      │
                                      └─ concat(x, coarse probability, t=0)
                                              │
                                              └─ refiner → predicted noise
                                                       │
                                                       └─ one-step correction

final logits = 0.5 × (coarse logits + corrected logits)
```

因此测试阶段不需要 label 或 layer curve；模型仍通过项目统一的 full-B-scan 3D sliding
window 执行推理。

### 3.5 关于 `inference_steps`

实验 YAML 当前包含：

```yaml
inference_steps: 8
```

但 `build_model()` 会移除该字段，当前 `DiffusionUNet3D` 也没有 iterative reverse
diffusion loop。也就是说，现有推理只在 `t=0` 执行一次确定性 correction，`8` 不会改变
推理次数或结果。

因此在论文或实验记录中，当前模型应描述为：

> conditional diffusion-style one-step mask refiner around a coarse 3D U-Net

不应描述为 8-step DDPM sampling。若未来加入多步反向采样，需要同步修改代码、实验配置、
速度/显存评估和本文档。

## 4. 两种结构对比

| 项目 | Baseline `UNet3D` | `DiffusionUNet3D` |
|---|---|---|
| 主干 | 1 个 3D U-Net | coarse U-Net + refiner U-Net |
| 当前 diffusion 实验输入 | OCT/OCTA 均可配置 | OCT + OCTA |
| 输出 | 直接 segmentation logits | coarse/refined logits 等权平均 |
| 训练条件 | image | image + noised target + timestep |
| 推理条件 | image | image + coarse probability + `t=0` |
| 辅助目标 | 无 | noise-prediction MSE |
| 推理随机性 | 无 | 无，当前为确定性 one-step refinement |
| 多任务输出 | 支持 `MultiTaskUNet3D` | 当前不支持 |
| 默认参数量（OCT+OCTA） | 5,735,361 | 7,171,042 |

Diffusion 模型的 7,171,042 个参数包括：

- coarse residual U-Net：5,735,361；
- refiner residual U-Net：1,435,681。

Baseline 参数量随输入通道略有变化：

| Input | Parameters |
|---|---:|
| OCT | 5,734,913 |
| OCT + OCTA | 5,735,361 |
| OCT + OCTA + large vessel | 5,735,809 |

## 5. 实验配置对应关系

| Config | Model | Input |
|---|---|---|
| `configs/experiments/unet3d_oct.yaml` | `UNet3D` | OCT |
| `configs/experiments/unet3d_oct_octa.yaml` | `UNet3D` | OCT + OCTA |
| `configs/experiments/unet3d_multimodal.yaml` | `UNet3D` | OCT + OCTA + large vessel |
| `configs/experiments/unet3d_multimodal_vasculature.yaml` | `MultiTaskUNet3D` | OCT + OCTA + large vessel |
| `configs/experiments/diffusion_unet_oct_octa.yaml` | `DiffusionUNet3D` | OCT + OCTA |

所有实验共享相同的数据发现、ROI/patch extraction、percentile normalization、balanced
sampling、validation metrics 和 sliding-window inference framework，因此比较 U-Net 与
U-Net + diffusion 时，主要改变的是模型结构和 diffusion auxiliary loss。
