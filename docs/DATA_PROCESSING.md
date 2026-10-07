# RNV segmentation 数据处理流程

本文档是本项目数据处理流程的完整说明，覆盖数据发现、训练/验证裁剪、输入模态构建、数据增强、预览和测试推理。实现的主要来源是 `configs/base.yaml`、`src/rnv_seg/data/`、`src/rnv_seg/transforms/` 和 `src/rnv_seg/engine/inference.py`。

## 文档维护规则

凡是修改数据发现、数据拆分、变量名、轴顺序、归一化、ROI、patch、采样比例、数据增强、输入模态、preview、validation 或 inference 的代码与配置，必须在同一次修改中同步更新：

1. 根目录 `README.md`；
2. 本文档 `docs/DATA_PROCESSING.md`；
3. 相关自动化测试。

若本文档与运行时 resolved config 不一致，以 resolved config 和代码为执行依据，但应立即修正文档。

## 1. 数据目录与文件发现

数据根目录由 `RNV_DATASET_ROOT` 指定。默认 split 和文件结构为：

```text
RNV_DATASET_ROOT/
  train_data/<scan_folder>/*_oct_octa.mat
  train_data/<scan_folder>/*_mask.mat
  train_data/<scan_folder>/*_seg.mat
  test_data/<scan_folder>/*_oct_octa.mat
  train_artery_vein_mask/<scan_folder>_inner_octa_mask.png
  test_artery_vein_mask/<scan_folder>_inner_octa_mask.png
```

严格发现模式要求每个训练 scan 恰好包含一个 `*_oct_octa.mat` 和一个 `*_mask.mat`。曲线 ROI 启用且 `required: true` 时还必须存在 `*_seg.mat`。`sample_id` 是 scan 文件夹名称的 SHA-256 前 16 位，用于 manifest 和程序内部标识；训练 patch 索引进度使用解析后的 lowercase case ID（如 `wfdr036`），不显示完整 scan 文件夹名称。

## 2. 数据拆分

若未指定独立 validation 文件夹，则从 `train_data` 按 subject 分组并执行多因素分层拆分：

- validation 比例：0.2；
- seed：42；
- 同一 subject 的所有 scan 必须位于同一 split；
- 对每个完整 structural RNV mask 做 26 邻域 3D 连通域分析，忽略小于 30 voxels 的
  component；以最大 component 与冻结阈值 36,586 voxels 比较，将 scan 分为
  `small_only` 或 `large_containing`，完全无 RNV 的 scan 标记为 `rnv_negative`；该阈值是
  当前 112 个阳性训练 scan 的最大 component 中位数，约等于 `0.0247 mm3`，训练、验证和
  checkpoint 选择始终共用同一个值；
- 从文件夹提取 `F120/F122/F124/F128/F130/F132` 等 acquisition field code；目前数据未提供
  field code 到 Superior/Inferior/Temporal/Nasal 的可信映射，因此只平衡真实 code，不猜测
  解剖名称；提供验证过的映射后才应在报告中换成四象限名称；
- 固定抽取约 20% subjects，在 10000 个可复现候选组合中最小化 field code、RNV presence、
  RNV group 和 scan 数量的分布误差；当某 field/group 出现在至少 2 个 subjects 时，缺失于
  train 或 validation 会受到覆盖惩罚；
- 若数据中存在完全 negative scan，默认要求 validation 至少包含一个；
- 每次运行把 split manifest 保存到对应 run 目录；
- manifest 每个 scan 记录 `field_code, rnv_voxels, largest_rnv_component_voxels,
  rnv_component_count, rnv_group`，并在 `summary` 中汇总 train/validation 的 scan 数、
  subject 数、field code 和 RNV group 分布；
- 发现 subject 泄漏时立即报错。

当前 split 版本为 `v3_subject_lesion_size_stratified`。因为拆分必须读取 113 个完整 mask，启动时会显示
`Indexing validation strata` 进度条。可在正式训练前单独生成和检查 manifest：

```bash
python scripts/create_split.py \
  --config configs/experiments/unet3d_oct.yaml \
  --output outputs/splits/rnv_v3_seed42.json
```

## 3. 数组轴、变量和物理间距

MATLAB 体积在代码中的统一轴顺序是：

```text
[depth, width, B-scan]
```

默认变量：

| 内容 | MATLAB 变量 |
|---|---|
| OCT | `imgMat` |
| OCTA | `imgMatAg` |
| RNV label | `mask` |
| RNV vasculature label（双输出实验） | `vasculature_mask` |
| 分层曲线 | `ManualCurveData` |

`unet3d_multimodal_vasculature.yaml` 要求同一个 `*_mask.mat` 同时包含 `mask` 和
`vasculature_mask`。两者均为 `[depth,width,B-scan]` 三维二值数组且 shape 必须一致；
`vasculature_mask` 必须是 `mask` 的子集。测试/推理阶段不读取这两个训练标签。

### 3.1 使用 structural RNV mask 生成 vascular pseudo-label

`vasculature_mask` 是由 structural `mask` 约束的 OCTA pseudo-label，不应在论文中描述为
人工标注 ground truth。处理严格限定在每个3D structural RNV 连通域内：

1. 只裁剪 structural mask 的局部包围盒，外扩 8 pixels，避免处理整幅空体积；
2. 对 OCTA 使用 `[0.8,0.6,0.6]` Gaussian sigma 去噪；
3. 使用 `[4,2,2]` Gaussian sigma 估计局部背景并计算正局部对比度；
4. 在 structural mask 内按第5和99百分位归一化强度与局部对比度；
5. score 为 `0.7 × OCTA intensity + 0.3 × local contrast`；
6. 每个 structural 连通域独立计算 Otsu 低阈值，并使用第85百分位作为高置信度种子；
7. 只保留与高置信度种子连通且至少5 voxels 的低阈值区域；
8. 最终强制执行 `vasculature_mask &= mask`。

Preview-only 命令不会修改数据：

```bash
python scripts/generate_vasculature_labels.py \
  --config configs/experiments/unet3d_multimodal_vasculature.yaml \
  --output-dir outputs/vasculature_label_preview \
  --max-scans 3
```

确认红色 structural / 青色 vascular overlay 后正式写入：

```bash
python scripts/generate_vasculature_labels.py \
  --config configs/experiments/unet3d_multimodal_vasculature.yaml \
  --output-dir outputs/vasculature_label_generation \
  --write-labels
```

每个原始 `*_mask.mat` 在替换前会备份到
`<output-dir>/original_target_backups/<scan_folder>/`。已有 `vasculature_mask` 默认跳过，
因此中断后可直接重复同一命令续跑；只有显式传入 `--overwrite` 才重新生成已有标签。

真实体素间距：

```yaml
voxel_spacing_mm: [0.003, 0.015, 0.015]
```

即 `[depth, width, B-scan] = [3, 15, 15] µm`，单体素体积为 `675 µm³ = 6.75×10⁻⁷ mm³`。

## 4. OCT 和 OCTA 强度处理

OCT/OCTA 根据 experiment 的 `data.modalities` 读取。训练和推理均先在每个完整原始
case、每个强度通道上计算 percentile statistics：

```yaml
normalize: percentile
percentile_range: [0.5, 99.5]
```

低于完整 case 的 0.5 percentile 的值截断为 0，高于 99.5 percentile 的值截断为
1，中间线性映射到 `[0,1]`。训练阶段先在未裁剪 volume 上计算 low/high，按
`(case_id, modality)` 在各 DataLoader worker 内缓存，再从原图裁剪 patch，并复用该
case 的 low/high。推理阶段直接对完整 volume 使用相同统计量。因此同一病例、同一
模态在训练 patch 和 full-volume inference 中具有完全一致的强度映射。输出固定为
`float32`。

## 5. Large-vessel 输入构建

Multimodal 网络只使用一个 large-vessel 通道，不再将 artery 和 vein 分开输入。RGB en-face mask 的处理顺序为：

1. RGB 转为 `[0,1]`；
2. artery 候选为 red 与 blue 通道的逐像素并集；
3. vein 候选为 green 与 blue 通道的逐像素并集；
4. 两者分别二值化，当前阈值为 `>0`；
5. 使用 8 邻域连通域，分别删除面积小于 30 pixels 的区域；
6. `large_vessel = cleaned_artery ∪ cleaned_vein`；
7. 要求 en-face mask 与体积 lateral plane 精确对齐；
8. 沿 depth 轴广播为 3D 通道；
9. Curved ROI 只参与 patch 定位；最终从未遮罩的原始 large-vessel volume 裁剪。

Multimodal 最终输入顺序为：

```text
data.modalities: [oct,octa,large_vessel]
network channels: [OCT, OCTA, large_vessel]
```

Large-vessel 始终保持严格二值 `{0,1}`，不接受 contrast 或噪声增强。

## 6. ILM/BM curved ROI

### 6.1 曲线读取

`ManualCurveData` 的形状预期为 `[width, layer, B-scan]`。当前使用：

- ILM：第 1 层；
- BM：第 8 层；
- MATLAB 坐标从 1 开始，载入后转换为 Python 0-based depth；
- 无效曲线点用 lateral plane 中最近的有效点补齐；
- 最后截断到合法 depth 范围。

### 6.2 上界：移动 ILM

训练/验证 scan 使用 RNV label 计算 ILM 的整体向上位移，同时保留原 ILM 曲线形状。

设最高 RNV depth 为 `z_top`，目标上界为：

```text
desired_upper = max(0, z_top - 20)
```

在所有位于 `z_top` 的 RNV lateral 坐标处读取 ILM，位移为：

```text
ilm_shift = max(0, max(ILM_at_highest_RNV) - desired_upper)
shifted_ILM = clip(ILM - ilm_shift)
```

因此移动后的 ILM 在最高 RNV 位置至少向上保留 20 px。

### 6.3 下界：移动 BM

BM 曲线逐点向下扩展 100 px。代码使用 lower-exclusive 表示：

```text
shifted_BM_exclusive = clip(BM + 100 + 1)
```

### 6.4 Curved ROI 的用途

每个 lateral 坐标 `(w,b)` 的 ROI depth 满足：

```text
shifted_ILM[w,b] <= depth < shifted_BM_exclusive[w,b]
```

该 curved ROI 仅用于过滤有效 RNV 锚点、限制候选 depth 范围和确定矩形 patch 的裁剪坐标。坐标确定后，程序直接从未进行 ROI masking 的原始 OCT、OCTA、large-vessel 和 label 中裁剪完整矩形 patch；不会把 patch 内、curved ROI 外的像素设为 0。

这样可以保留 patch 内真实的组织和玻璃体背景，也避免 Gaussian noise 只出现在 ROI 内而形成不均匀或可被模型识别的人工边界。

## 7. RNV 导向 patch 索引

共享训练 patch 为：

```yaml
patch_size: [480, 224, 24]
```

索引流程：

1. 只保留 curved ROI 内的 RNV 体素；
2. 使用 26 邻域识别 3D RNV 连通域；
3. 忽略小于 30 voxels 的 RNV 连通域；
4. 每个有效连通域至少产生一个前景锚点；
5. 大于 patch 覆盖范围的病灶按 0.5 overlap 产生多个锚点；
6. 同一个 `480×224×24` 窗口可以覆盖的相邻锚点贪心合并；
7. 无法由同一窗口覆盖的远距离或多 B-scan RNV 自动移动窗口；
8. 若窗口不能消耗当前锚点，立即报错，禁止无限循环；
9. 训练集对每个基准窗口增加 width 左/右和 B-scan 前/后平移视角；移动量分别为
   `224×0.25=56 px` 和 `24×0.25=6 frames`，并可加入对角视角；depth start 不随机移动；
10. 基准窗口覆盖病灶至少 80% 时标记为 `positive_complete`；平移窗口必须保留至少
    20% 病灶并标记为 `positive_boundary`，同时必须含有至少 30 个 RNV voxels；
11. 边界截断后坐标相同的窗口去重；普通上限为每个 component 6 个、每个 case 20 个，
    但覆盖长病灶或多个病灶所必需的基准窗口不会因上限被删除；
12. 每个 epoch 的 positive slot 同时满足两级配额：scan group 按
    `small_only/large_containing = 70%/30%`，view type 按 complete/boundary = 55%/45%；
    group 来自裁剪前完整 GT 的最大3D component，既不作为网络输入，也不改变二值 label；随后按
    case → lesion round-robin；同一个 3D lesion 每个 epoch 最多使用 2 个 views，其余 shifted
    views 跨 epoch 轮换，避免单个病灶的高度相似裁剪被反复学习；
13. 这些窗口只用于 training；validation 直接使用完整 scan，不再读取 patch index；
14. 再在 rectangular ROI 的每个无重叠 B-scan 槽位中生成 depth 上/中/下和 width
    左/中/右 negative 候选；空间 phase 为 `[0.0,0.5,1.0]`；
15. Negative 候选必须满足 structural RNV voxel 数严格等于 0；任何包含 RNV 的候选均
    从 negative 池剔除；
16. 不再把 scan 几何中心或视网膜厚度当作黄斑定位依据，因为 temporal retina 同样可能
    很薄。纯 negative 只分为 RNV 扩展邻域内的 `adjacent` 和全 scan 空间覆盖的
    `anatomical_coverage`，普通抽样比例为 30%/70%。后者是中性覆盖标签，不声称已经识别
    黄斑、视盘或颞侧位置；候选在每个 scan 的全 width、全 B-scan 槽位及多个 spatial phase
    中轮换，因此数据中实际存在的 macula、disc、temporal 区域均有机会被抽到；
17. ordinary negative 按 scan record round-robin，避免候选较多的 Superior/Inferior、
    Temporal/Nasal 扫描块主导训练；每个 epoch 的同一 B-scan 槽位最多选择一个 negative，
    因此被选中的第二阶段 negative patch 彼此不重叠；不同 epoch 可轮换槽位内的空间 phase；
18. 每个物理 microbatch 强制 positive:negative = 1:1。当前 physical batch size 为 2，
    即每个 microbatch 含 1 个 RNV positive 和 1 个纯 negative；连续 2 个 microbatch
    累积梯度后再更新参数，因此 effective batch size 4（合计 2 positive + 2 negative）。
    模型参数保持 float32，CUDA autocast 降低卷积 activation 精度和显存；Dice/BCE/Focal/
    Tversky 的 sigmoid 与归约强制使用 float32，避免大体积 FP16 求和溢出。B-scan patch
    从 32 降到 24 帧，使每个 patch 的 activation voxel 数再减少 25%。

双输出实验同样只使用 structural `rnv` mask 生成前景导向窗口；`vasculature_mask`
仍作为第二个监督目标随同一窗口裁剪，但不再单独改变采样位置。

### 7.1 Hard-negative mining

前 8 epochs 使用普通分层 negative。自第 8 epoch 起每 2 epochs：

1. 每轮最多评估 512 个候选，使用 AMP batch size 1；25% 名额循环复查已有 hard pool，
   75% 名额按 adjacent/anatomical_coverage 配比、病例及空间槽位 round-robin 发现新错误；
2. 关闭随机增强并用当前模型的 inference forward 计算 RNV probability；
3. 以 `probability >= 0.4` 产生预测 mask，并应用与测试相同的小区域和连续 B-scan
   后处理；只有后处理后仍至少包含 20 个预测 RNV voxels 的 patch 才进入 hard pool。
   相比最终推理阈值提前纳入 near-miss，增加正常组织与 RNV 的分类 margin，同时仍排除
   最终会被测试后处理删除的孤立噪点；
4. score = 最高 4096 个保留 voxel 的平均 probability + `0.35 ×` 对数归一化的 FP
   voxel extent，使高置信、小范围错误和较大范围持续假阳性都能获得优先级；
5. 同一 anatomical slot 的不同 spatial phase 只保留分数最高者；每个 case 最多保留
   16 个，整个池最多 768 个，避免少数病例或重复位置占满 hard pool；
6. 未在本轮重新确认的旧 score 每轮乘 `0.9`，低于 `0.35` 后移除；重新检查后变干净的
   patch 立即删除，因此早期模型产生但后期已修复的错误不会永久占池；
7. 训练抽取 hard negative 时按 case round-robin，并在每个 case 内按 score 排序；hard
   negatives 最多替换 50% 的普通 negative，batch 总体仍保持
   positive:negative = 1:1；
8. hard-negative pool 保存到 checkpoint 的 `training_state`，断点续训可恢复；这些内部
   状态不写入 `logs/metrics.jsonl`，也不增加 terminal 的常规监控字段。

为解决 patch 候选未必覆盖完整 scan 实际假阳性的问题，`patch_plus_full_volume_v2` 另外从
第 8 epoch 开始每 4 epochs 轮换选择 4 个**训练集**完整 scans：使用测试同款完整 B-scan
滑窗、`probability >= 0.5` 和后处理找到持续假阳性 component，再把它们映射回已经确认
不含 RNV label 的 pure-negative patch 坐标并更新同一个 hard pool。每轮病例顺序由固定
seed 决定并循环覆盖。该过程不能访问 validation records，避免把验证错误反馈进训练。

当前 Focal Tversky 设置为 `alpha=0.6, beta=0.4`。实现中 alpha 乘 false positive，
因此该设置比 false negative 更强地惩罚正常黄斑误检。

### 7.2 Hard-positive mining

为提高 small-only、病灶边界和低对比度 RNV 的置信度，从第 3 epoch 开始，每个 epoch
训练和验证结束后以 AMP batch size 1 评估最多 256 个**训练集 positive patches**，
本轮结果从下一 epoch 起生效：

1. 关闭随机增强，使用当前模型得到 RNV sigmoid probability；
2. 在真实 RNV label 内计算 `probability < 0.45` 的 missed voxels；至少漏掉 10 voxels 且
   missed fraction 至少为 5% 才能进入 hard-positive pool；
3. 以真实 RNV bounding box 向各方向扩展 4 voxels，并用框内非 RNV context 与 RNV 的
   OCT 均值差/局部标准差计算对比度；对比度越低，`low_contrast` 分数越高；不使用 OCTA
   或 vessel prior 判断低对比度；
4. 基础 score 为 `0.5 × missed_fraction + 0.3 × normalized confidence shortfall +
   0.2 × low_contrast`；`small_only` 额外加 0.15，`positive_boundary` 额外加 0.10；
5. 每轮候选预算的 50% 复查旧 hard positives，另外 50% 按 case/lesion round-robin
   发现新候选；pool 最多 256 个 patch、每 case 12 个、每 lesion 3 个；未复查的旧分数
   每轮乘 0.9，低于 0.15 后删除；
6. 下一 epoch 最多 30% positive slots 优先从 hard-positive pool 选择；困难 lesion 允许
   最多 3 个增强 views，而普通 lesion 仍最多 2 个。该 30% 是 positive 内部重分配，
   不改变 microbatch 的 positive:negative = 1:1；
7. hard-positive pool 保存到 checkpoint `training_state` 并在 resume 时恢复；不写入精简的
   `metrics.jsonl`，也不增加常规 terminal 监控字段。

GT 分组和低对比度分数只控制训练 patch 的重采样，不作为模型输入，也不把 RNV 改成多类别。
Validation/test 从不进入 hard-positive pool，避免验证信息泄漏。

Depth 方向不再把 RNV 锚点放在 patch 中心。当前配置为：

```yaml
rnv_depth_position_fraction: 0.40
```

Depth start 的基本计算为：

```text
depth_start = rnv_anchor_depth - round(480 × 0.40)
```

因此 RNV 锚点通常位于 patch 顶部以下约 192 px，而不是中心的 240 px；相当于比 `0.30` 设置向上移动约 48 px，同时仍比完全居中裁剪向下约 48 px。这样可增加病灶上方玻璃体覆盖，并保留约 288 px 的 RNV 下方视网膜/BM 区域。若接近 ROI 或原体积边界，start 会自动截断到合法范围。Width 和 B-scan 方向仍以 RNV 锚点为中心，并可合并同一窗口覆盖的多个病灶。

训练启动时逐 scan 输出窗口数量和 ILM shift，例如：

```text
Indexing rnv patches [1/1] wfdr017: 5 positive views (complete=1, boundary=4) + <negative_total> pure-negative candidates (adjacent=<n>, anatomical_coverage=<n>), ILM shift=34px
Indexed <positive_total> multi-view foreground-guided, <negative_total> pure-negative candidates; balanced samplers select positive:negative=1:1 per epoch
```

## 8. 训练数据增强

增强只在 training dataset 执行，validation 和 test 不执行随机强度增强。

### 8.1 随机翻转

```yaml
random_flip_axes: [1, 2]
random_flip_probability: 0.5
```

仅沿 width 和 B-scan 方向翻转，不沿 depth 翻转，以保持玻璃体—视网膜的深度方向。

### 8.2 Contrast

对 OCT/OCTA 每个通道独立处理：

```yaml
probability: 0.5
range: [0.8, 1.2]
```

以 curved ROI 内该通道均值为中心随机缩放 contrast，随后截断到 `[0,1]`。

### 8.3 Gaussian noise

对 OCT/OCTA 每个通道独立处理：

```yaml
probability: 0.3
std_range: [0.0, 0.05]
```

每次随机选择噪声标准差并在完整矩形 OCT/OCTA patch 上加入零均值高斯噪声，随后截断到 `[0,1]`。不会使用 curved ROI mask 限制噪声区域。Label 与 large-vessel 不进行强度增强。

## 9. 最终训练张量

当前典型输入：

| Experiment | 输入 shape |
|---|---|
| OCT | `[1,480,224,24]` |
| OCT + OCTA | `[2,480,224,24]` |
| Multimodal | `[3,480,224,24]` |

单输出 Label shape 为 `[1,480,224,24]`；双输出为 `[2,480,224,24]`，通道顺序固定为
`[rnv, vasculature]`。dtype 均为 `float32`，取值为 `{0,1}`。空间翻转对两个标签同步执行；
contrast 与 Gaussian noise 仍只作用于 OCT/OCTA。

## 10. Validation

Validation 不再构建或运行 ROI patch dataset。每个 epoch 都逐个处理完整 validation scans，
模型输入不包含 curve 或 label；validation loader 读取完整 label 只用于 loss 和指标计算。
Validation 不执行随机 contrast、Gaussian noise、flip 或 crop jitter。

训练 terminal 使用精简监控。Training batch 进度条只显示 running loss 和 RNV Dice；
validation 进度条只显示 validation loss 和 RNV Dice。每个 epoch 的最终汇总行固定只输出：

```text
train_loss, train_dice, val_loss, val_dice, lr, is_best_loss, is_best_dice
```

新最低 full-volume validation loss 使用 `[BEST LOSS]` 和 `[BEST LOSS MODEL]`；
`[BEST MODEL]` 表示当前 epoch 更新了最终推荐的最低-loss checkpoint，其他
epoch 使用 `[NOT BEST]`。Lesion、volume、negative
false-positive、GPU peak、耗时、checkpoint 状态及 LR 是否下降等详细字段不再展开到 terminal，
不展开到 terminal；`last.pt` 每个 epoch 保存。分组诊断会写入 JSONL，但不会增加 terminal 噪声。
`logs/metrics.jsonl` 每个 epoch 只保留：

```text
epoch, train_loss, train_dice, val_loss, val_dice, positive_dice,
small_only_dice, large_containing_dice, stratified_rnv_score,
negative_patch_clean_rate, lr, is_best, is_best_loss, is_best_dice,
best_epoch, best_loss_epoch, best_dice
```

其中 `train_dice` 是整个训练 epoch 累计的 RNV hard Dice；`val_dice` 是逐个完整 scan 经测试
同款后处理后的 RNV Dice 平均值；`small_only_dice` 与 `large_containing_dice` 分别在两组
完整 scan 内作 case mean。`best_epoch` 与 `best_loss_epoch` 都记录最低 validation loss
对应的 epoch，`best_dice` 记录同一 epoch 的总体 Dice。Stratified score 仅作为诊断，
不再控制 checkpoint、LR 或 early stopping。`positive_dice` 只对含 RNV 的完整 scans 求
平均；`negative_patch_clean_rate` 表示完全阴性 validation scans 中后处理预测为空的比例。
只有 validation 中缺少对应阳性或阴性类别时，该字段才记录为 JSON `null`。当前数据只有
一个完全阴性 validation scan，因此 clean rate 可用于发现问题，但统计稳定性有限。

### 10.1 Full-volume checkpoint selection

所有实验每个 epoch 都执行 full-volume validation，并继承以下配置：

```yaml
training:
  full_volume_checkpoint:
    enabled: true
    every_n_epochs: 1
    metric: validation_loss
    mode: min
    preferred_metric: validation_loss
    preferred_mode: min
    preferred_min_delta: 0.002
    selection_metric_weights:
      small_only_dice: 0.35
      small_only_lesion_f1: 0.30
      large_containing_dice: 0.25
      negative_clean_rate: 0.10
    evaluate_at_training_end: true
```

每个 epoch 逐个载入完整 validation scan。
模型输入只包括配置的真实输入模态，不读取 `ManualCurveData`；RNV label 不进入模型，仅用于
预测后的评分。
该流程与测试完全共用 percentile normalization、完整 B-scan 的 `[768,600,24]` 重叠滑窗、
`0.45` probability threshold 及后处理：先删除每帧小于 30 pixels 的区域，再要求同一 3D RNV component 至少连续
5 帧出现。Loss 使用重叠融合后的 raw logits 与完整 label 计算，不对后处理后的二值 mask
计算 loss；Dice 则使用后处理结果。每个 scan 先独立得到一个 loss 和一组指标，最终再对
case 做算术平均，不把所有 voxel 拼接后统一计算，因此较大的 scan 不会获得更高权重。

- `best.pt`：full-volume validation loss 最低，作为默认推理模型；
- `best_loss.pt`：与 `best.pt` 相同的最低-loss checkpoint；
- `best_dice.pt`：旧文件名兼容，当前同样保存最低-loss模型；
- `best_full_volume.pt`：旧文件名兼容，当前同样指向最低-loss模型；
- `last.pt`：最后完成 epoch，用于断点续训；
- `logs/full_volume_metrics.jsonl`：除 loss/Dice/checkpoint 状态外，还记录
  `small_only_dice, large_containing_dice, stratified_rnv_score`。

`checkpoint_metric=validation_loss`、`checkpoint_mode=min`、默认 `best.pt`、scheduler
monitor 和 early stopping 均使用同一个 full-volume case-mean validation loss；它们不受
Dice、size-stratified score 或 `multitask_score` 的短期波动影响。Small-only Dice、lesion
F1、large-containing Dice 和 negative clean rate 继续写入日志用于诊断，但不参与模型选择。
整卷验证计算量较大，因此每个 epoch 会明显变慢。

### 10.2 `best.pt` 阈值与后处理校准

训练完成后固定 `best.pt` 权重，只使用 validation labels 校准最终二值化和后处理参数：

```bash
python scripts/calibrate.py \
  --config outputs/<run>/resolved_config.yaml \
  --checkpoint outputs/<run>/checkpoints/best.pt \
  --output-dir outputs/<run>/calibration
```

默认扫描 60 种组合：

- sigmoid probability threshold：`[0.45,0.5,0.55,0.6,0.65]`；
- 每张 B-scan 的最小二维连通区域：`[5,10,20,30]` pixels；
- 同一 3D component 的最小连续 B-scan 数：`[3,5,7]`。

每个完整 validation scan 只执行一次网络推理；得到的 RNV probability 在内存中依次完成
24 组二值化与后处理，随后释放，不重复运行模型。`positive_dice` 是阳性 validation cases
的 case-mean Dice；lesion sensitivity、precision 和 F1 由所有 validation cases 的
detected/target/false-positive lesion 计数汇总得到。首先剔除 lesion sensitivity `<0.82`
的组合，再最大化 `0.5 × positive_dice + 0.5 × lesion_f1`，平局时优先 false-positive
lesions 更少的组合。

输出包括全部组合的 `calibration_results.csv`、选择依据与最佳无约束组合的
`calibration_summary.json`，以及可直接用于 evaluate/infer 的 `calibrated_config.yaml`。
若没有组合达到 sensitivity 下限，程序不会生成 calibrated config，避免静默部署低敏感度
参数。Test data 不参与校准，否则会造成评价泄漏；同一 validation set 同时用于 checkpoint
选择和校准时结果仍可能略乐观，正式报告最好使用独立 calibration subset 或 nested CV。

### 10.3 Validation-loss plateau learning rate

Learning rate 使用 `ReduceLROnPlateau` 监控 `validation_loss`：

```yaml
scheduler:
  name: reduce_on_plateau
  monitor: validation_loss
  mode: min
  factor: 0.5
  patience: 5
  smoothing_window: 3
  threshold: 2.0e-3
  threshold_mode: abs
  cooldown: 0
  minimum_learning_rate: 1.0e-6
```

先对最近 3 个 epochs 的 validation loss 求移动平均。当平滑值连续 5 epochs 没有超过
绝对阈值 `0.002` 的改善时，learning rate
乘以 0.5，最低不小于 `1e-6`。Terminal 只在 epoch 汇总的 `lr` 字段显示当前值，不再额外
打印 LR reduction 详情；当前 learning rate 同时写入 metrics log 的 `lr` 字段。
Scheduler state 保存在所有 checkpoint 中，断点续训不会重新计算 patience。由于 checkpoint
目标已从 patch Dice 改为 full-volume validation loss，旧训练不能直接使用新配置 resume，
需要开始新的 run。
同样，`patch_plus_full_volume_v2` 会改变训练样本序列，不会静默注入旧 checkpoint。旧 run
若中断，应使用该 run 自己保存的 `resolved_config.yaml` 恢复；要评估新的 hard-negative
策略则启动新 run，避免混合两种采样协议后无法解释实验结果。

Early stopping 同样监控 full-volume validation loss，patience 为 18。该窗口允许 plateau
scheduler 先降低一次 LR，并继续观察多个低 LR epochs，同时避免在已出现泛化下降后继续
训练约 20 个高成本 full-volume epochs。

## 11. Preview 输出

训练开始时默认输出 4 个增强后的训练样本，保存 `sample.npz`、`metadata.json` 和模态对应图片：

- 3 个 positive previews：优先来自不同 scan/case 和不同 3D RNV component；同一 component
  有多个 shifted/boundary views 时只选一个代表，优先展示覆盖完整病灶的 `positive_complete`；
- 1 个 pure-negative preview：优先选择 `anatomical_coverage`，再考虑 `adjacent`，
  并尽量避开已用于 positive preview 的病例。

Preview 不再直接读取 dataset 的前 3 个连续 indices，因为这些 indices 通常是第一个病例同一
RNV component 的多个平移视角。每个 `metadata.json` 额外记录 `case_id`、原始
`preview_dataset_index`、`patch_start`、`lesion.component_id` 和 `sampling_type`，用于确认
病例、病灶和裁剪坐标确实不同。该选择只影响 QA preview，不改变训练 batch sampler。

| 输入模态 | Preview 图片 |
|---|---|
| OCT | `input_oct_bscans.png`, `label_bscans.png`, `overlay_bscans.png` |
| OCT + OCTA | 上述文件加 `input_octa_bscans.png` |
| Multimodal | 上述文件加 `input_large_vessel_enface.png` |

双输出 preview 另外保存 `label_rnv_bscans.png`、`overlay_rnv_bscans.png`、
`label_vasculature_bscans.png` 和 `overlay_vasculature_bscans.png`。

展示的 B-scan 优先选择 RNV label 体素最多的切片。Overlay 使用 OCT 灰度图和红色 RNV label。

手动生成命令：

```bash
python scripts/preview_data.py \
  --config configs/experiments/unet3d_multimodal.yaml \
  --output-dir outputs/data_preview \
  --samples 3
```

## 12. Test / inference

正式测试严格遵循此前约束：

- 不读取 `ManualCurveData`；
- 不读取 RNV label 进行定位；
- 输入完整 B-scan 高度和宽度；
- inference patch 为 `[768,600,24]`；
- 只需要沿 B-scan 序列滑动；
- overlap 为 0.5；
- 重叠区域 logits 求平均；
- 累积 logits 保存在 CPU，控制 GPU 显存；
- sigmoid threshold 为 0.45。

Threshold 后执行以下后处理：

1. 对每张 B-scan 的 `[depth,width]` 二值 mask 使用 8 邻域连通域；
2. 删除面积小于 30 pixels 的二维区域，即只保留 `area >= 30`；
3. 对清理后的完整体积使用 26 邻域三维连通域识别同一个 RNV；
4. 对每个三维 RNV 统计其出现的 B-scan index；
5. 只有最长连续 B-scan 序列至少为 5 帧时才保留；总共出现 5 帧但中间断帧不满足要求。

双输出实验对 vasculature 使用独立的温和后处理：删除小于 10 voxels 的 3D 连通域，
要求至少连续 2 个 B-scan，随后执行 `vasculature_mask &= mask` 以满足解剖层级约束。

每个 case 单独保存到以原始 scan 文件夹名作为 case ID 的目录：

```text
predicts/<case_id>/<source_file_id>_mask.mat
```

输出为与 label 兼容的 MATLAB 文件：

```text
variable: mask
shape: [768,600,600]
dtype: uint8
values: {0,1}
```

双输出 MAT 在同一文件中同时保存 `mask` 与 `vasculature_mask`，shape 和 dtype 相同。
完整 float32 sigmoid probability volume 默认不写入 MAT，以降低 host memory、MAT 文件大小
和保存时间。需要精确重新校准 threshold 时可设置 `inference.save_probabilities: true`；此时
RNV 变量为 `mask_probability`，双输出实验另有 `vasculature_mask_probability`。

每个 case 文件夹还为每个输出头保存以下概率诊断文件：

```text
<head>_probability_enface.png
<head>_probability_enface_overlay.png
<head>_probability_bscans.png
<head>_probability_bscan_overlay.png
<head>_probability_heatmap.json
```

En-face 图使用 depth 方向最大概率投影；overlay 优先使用 OCTA en-face 背景，没有 OCTA
时使用 OCT。B-scan montage 默认选择 probability 总和最高的 6 帧并叠加到 OCT 上。
JSON 同时记录 Python 0-based 和 MATLAB 1-based B-scan index、二值化 threshold、概率范围及
超过 threshold 的 voxel 数。所有 PNG 使用固定的 0–1 连续色标，显示的是后处理前概率；
因此可以区分“模型概率低于 0.45”与“概率已经过阈值、但被 30-pixel/连续 5 帧后处理删除”这
两类漏检。`probability_heatmaps.bscan_count` 和 `overlay_alpha` 可在 inference 配置中调整。

## 13. 当前关键默认值

当前 U-Net 恢复为 24 base channels，并使用 dropout 0.1、AdamW weight decay
`1e-5`；16-channel 强正则化消融在相同 full-volume split 上表现较差。该变化会改变
checkpoint tensor shape，因此 16-channel `last.pt` 不能用当前实验 YAML 续训；新方案必须
开启新 run。校准或推理旧 `best.pt` 时，应使用该 run 中保存的
`resolved_config.yaml` 重建原模型。

| 项目 | 当前值 |
|---|---|
| 训练 patch | `[480,224,24]` |
| 推理 patch | `[768,600,24]` |
| 默认预测 threshold | `0.45` |
| Probability 输出 | en-face/B-scan 热图默认开启；完整 MAT float32 概率默认关闭 |
| Training dtype / AMP | float32 master weights and inputs / CUDA AMP enabled |
| Model regularization | base channels 24；dropout 0.1；AdamW weight decay `1e-5` |
| Physical / effective batch size | 2 / 4（每次参数更新合计 2 positive + 2 negative） |
| Validation | 每个 epoch 完整 scan；测试同款 `[768,600,24]` overlap sliding windows |
| LR / early stopping monitor | full-volume `validation_loss`（min） |
| Checkpoints | `best.pt`/`best_loss.pt` 均为最低 full-volume validation loss；其他 best 文件名仅兼容 |
| RNV 导向 patch | 启用；覆盖所有合格 structural RNV 连通域，不设固定比例 |
| Train batch positive:negative | 1:1 |
| 每个 RNV component positive views | 基准 + width/B-scan 平移；普通上限 6 |
| 每个 case positive views | 普通上限 20；必要基准窗口不删除 |
| Positive complete/boundary 抽样 | 0.55 / 0.45 |
| Positive scan group 抽样 | small-only / large-containing = 0.7 / 0.3；阈值 36,586 voxels |
| 每个 lesion 每 epoch positive 上限 | 2 views；其余 views 跨 epoch 轮换 |
| Positive 平移 | width ±56 px；B-scan ±6 frames；depth 不移动 |
| Positive component coverage | complete ≥0.8；boundary ≥0.2 |
| 普通 negative 分类比例 | adjacent/anatomical_coverage = 0.3/0.7 |
| 解剖区域覆盖 | 不使用 thickness 或 scan 中心定位 macula；按 scan、B-scan 槽位和 spatial phase 轮换 |
| Negative overlap | 同一 epoch 每个 B-scan 槽位最多一个，彼此不重叠 |
| Hard-negative mining | epoch 8 开始，每 2 epochs，最多 512 candidates；测试同款后处理后至少 20 FP voxels；每 case ≤16、总池 ≤768；旧 score ×0.9 |
| Hard-negative 占 negative 上限 | 0.5 |
| Hard-positive mining | epoch 3 开始，每 epoch 最多 256 candidates；GT 内 probability <0.45；结合漏检比例、置信度缺口和 OCT 低对比度 |
| Hard-positive 占 positive 上限 | 0.3；困难 lesion 每 epoch 最多 3 views |
| Focal Tversky alpha/beta | 0.6 / 0.4 |
| LR scheduler | ReduceLROnPlateau，monitor=`validation_loss` |
| LR plateau patience/factor/min | 5 epochs / 0.5 / `1e-6` |
| ILM layer | 1 |
| BM layer | 8 |
| RNV 上方 margin | 20 px |
| RNV 在 depth patch 中的位置 | 0.40（约第 192 px） |
| BM 下方 margin | 100 px |
| RNV 连接性/最小体素 | 26 / 30 |
| Large-vessel 连接性/最小面积 | 8 / 30 pixels |
| Sliding-window overlap | 0.5 |
| 每帧最小区域 | 30 pixels |
| RNV 最小连续帧数 | 5 B-scans |
| `best.pt` calibration grid | threshold `[0.45,0.5,0.55,0.6,0.65]` × area `[5,10,20,30]` × consecutive `[3,5,7]`；sensitivity ≥0.82 |
| Voxel spacing | `[0.003,0.015,0.015] mm` |

## 14. 相关实现文件

| 功能 | 文件 |
|---|---|
| 共享配置 | `configs/base.yaml` |
| Scan 发现 | `src/rnv_seg/data/discovery.py` |
| Train/validation split | `src/rnv_seg/data/splits.py` |
| 模态构建和 dataset | `src/rnv_seg/data/dataset.py` |
| Balanced batch / hard-negative selection | `src/rnv_seg/data/sampling.py` |
| ILM/BM ROI 与病灶锚点 | `src/rnv_seg/data/roi.py` |
| Flip/contrast/noise | `src/rnv_seg/transforms/patches.py` |
| Preview | `src/rnv_seg/data/preview.py` |
| Full-volume sliding window | `src/rnv_seg/engine/inference.py` |
| 工作流入口 | `src/rnv_seg/workflows.py` |
