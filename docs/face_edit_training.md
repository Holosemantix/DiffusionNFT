# 人脸编辑 SFT/RL 训练完整指南

> 本文档详细说明 DiffusionNFT 分支中“小脸低清修复 / 人脸编辑”的完整训练流程、数据构造、启动命令与代码地图。
> **当前推荐主线**：`Flux2 Edit SFT → Flux2 Edit RL`；SD3 路径保留为备选与对照实验。

---

## 目录

- [1. 任务定义与整体训练阶段](#1-任务定义与整体训练阶段)
- [2. 环境准备](#2-环境准备)
- [3. 数据集处理详解](#3-数据集处理详解)
  - [3.1 三种数据源与处理方式](#31-三种数据源与处理方式)
  - [3.2 WIDER-FACE 小脸修复的合成逻辑](#32-wider-face-小脸修复的合成逻辑)
  - [3.3 MagicBrush 通用编辑数据](#33-magicbrush-通用编辑数据)
  - [3.4 canonical JSONL 统一格式](#34-canonical-jsonl-统一格式)
  - [3.5 物化数据：从在线数据集到本地 JSONL](#35-物化数据从在线数据集到本地-jsonl)
  - [3.6 快速数据检查](#36-快速数据检查)
- [4. Flux2 训练路径](#4-flux2-训练路径)
  - [4.1 Stage 0：T2I SFT（冒烟测试）](#41-stage-0t2i-sft冒烟测试)
  - [4.2 Stage 1：Edit SFT（图像编辑监督微调）](#42-stage-1edit-sft图像编辑监督微调)
  - [4.3 Stage 2：Edit RL / DiffusionNFT（强化学习优化）](#43-stage-2edit-rldiffusionnft强化学习优化)
  - [4.4 Flux2 采样验证](#44-flux2-采样验证)
- [5. SD3 训练路径](#5-sd3-训练路径)
  - [5.1 SD3 Edit SFT](#51-sd3-edit-sft)
  - [5.2 SD3 Edit RL / DiffusionNFT](#52-sd3-edit-rldiffusionnft)
- [6. 人脸奖励（Reward）与损失（Loss）详解](#6-人脸奖励reward与损失loss详解)
  - [6.1 SFT 阶段损失](#61-sft-阶段损失)
  - [6.2 RL 阶段奖励设计](#62-rl-阶段奖励设计)
  - [6.3 DiffusionNFT 前向过程损失](#63-diffusionnft-前向过程损失)
- [7. LoRA 输出路径与恢复训练](#7-lora-输出路径与恢复训练)
- [8. 主要代码地图](#8-主要代码地图)
  - [8.1 入口脚本层](#81-入口脚本层)
  - [8.2 人脸编辑核心模块层](#82-人脸编辑核心模块层)
  - [8.3 原始 DiffusionNFT / Flow-GRPO 模块层](#83-原始-diffusionnft--flow-grpo-模块层)
- [9. 常见问题与排查](#9-常见问题与排查)

---

## 1. 任务定义与整体训练阶段

### 任务定义

本分支针对的核心任务是：**输入一张带有低分辨率或退化小脸的 `source_image`，根据文本指令修复人脸区域，输出 `target_image`，同时尽量保持非编辑区域不变。**

统一的数据契约（data contract）为：

```text
source_image + instruction + mask_image → target_image
```

各字段含义：

| 字段 | 说明 |
|------|------|
| `source_image` | 编辑前图片。WIDER-FACE 路径中是被合成退化过的人脸图。 |
| `target_image` | 编辑后目标图。WIDER-FACE 路径中是原始清晰图。 |
| `mask_image` | 单通道 mask，**白色**表示可编辑/重点评估区域，**黑色**表示应保持区域。 |
| `instruction` | 编辑文本指令。WIDER-FACE 默认指令为 `restore the small low-resolution face while preserving the rest of the image`。 |

### 三阶段训练流程

```
┌─────────────────┐     ┌─────────────────┐     ┌─────────────────┐
│  Stage 0: T2I   │     │  Stage 1: Edit  │     │  Stage 2: Edit  │
│      SFT        │ ──→ │      SFT        │ ──→ │      RL         │
│  (冒烟测试)      │     │  (编辑能力)      │     │  (人脸奖励优化)  │
└─────────────────┘     └─────────────────┘     └─────────────────┘
```

| 阶段 | 目的 | 输入 | 监督/优化目标 |
|------|------|------|---------------|
| **T2I SFT** | 验证 Flux2 基础文生图训练链路是否通畅 | `instruction + target_image` | 标准 flow-matching 损失 |
| **Edit SFT** | 让模型学会“参考 source 图、按指令编辑 target 区域” | `source_image + instruction + mask_image + target_image` | 只对 target tokens 做 flow matching；mask 区域加权；背景做 preserve loss |
| **Edit RL** | 用旧策略采样多个候选，用人脸奖励打分，通过 DiffusionNFT 前向过程损失继续优化 LoRA | `source_image + instruction + mask_image` | 人脸奖励（detection / identity / landmark / preserve / target）→ advantage → DiffusionNFT forward loss |

> **建议顺序**：必须按 `T2I SFT → Edit SFT → Edit RL` 顺序执行。Edit RL 必须加载 Edit SFT 产出的 LoRA 作为起点。

---

## 2. 环境准备

### 基础依赖

```bash
conda create -n DiffusionNFT python=3.10.16
conda activate DiffusionNFT
pip install torch==2.6.0 torchvision==0.21.0 --index-url https://download.pytorch.org/whl/cu126
pip install -e .
```

### Flux2 人脸训练额外依赖

```bash
pip install -e ".[flux2,face]"
```

如果 `flux2` 无法通过 pip 安装，可手动 clone BFL 官方仓库后使用 `PYTHONPATH=/path/to/flux2/src` 启动脚本。

模型与数据集来自 Hugging Face，受限模型或私有缓存环境需提前执行：

```bash
huggingface-cli login
```

---

## 3. 数据集处理详解

### 3.1 四种数据源与处理方式

所有 SFT/RL 脚本均通过 `flow_grpo.editing_data.FaceEditDataset` 读取数据，`--dataset_source` 支持以下值：

| `dataset_source` | 用途 | 数据处理方式 |
|------------------|------|--------------|
| `wider_face_restore` | **推荐**的人脸修复起步数据 | 从 `CUHK-CSE/wider_face` 读取图片与人脸框，筛选小脸框，对框内区域做低清/JPEG/模糊退化；原图作为 target，退化图作为 source。 |
| `face_aug_preserve` | 只有含脸图片、没有编辑对时的保 ID 合成数据 | 从本地 `--image_dir` 读普通图片，用 OpenCV Haar 检测人脸并过滤无脸图；随机生成小脸复原、非脸区域背景增强、no-op 保持三类样本，指令中加入 `[preserve face id]` 等触发词。 |
| `magicbrush` | 通用图像编辑数据 | 从 `osunlp/MagicBrush` 读取真实的 `source_img / instruction / target_img / mask_img` 四元组。 |
| `canonical_jsonl` | 已落盘的可复现实验数据 | 从 JSONL 读取统一字段，图片路径相对 JSONL 所在目录，也可使用绝对路径。 |

### 3.2 WIDER-FACE 小脸修复的合成逻辑

合成逻辑位于 `flow_grpo/editing_data.py`，关键步骤如下：

1. **框提取** (`_extract_boxes`)：兼容 `faces.bbox`、`annotations.bbox`、`bbox` 等 WIDER-FACE 镜像字段格式。
2. **小脸筛选** (`_select_small_boxes`)：默认选择相对图像尺寸较小的人脸，`small_face_fraction=0.12`，单图最多 `max_faces_per_image=8` 个。
3. **退化合成** (`degrade_face_regions`)：对选中的人脸框区域做以下退化：
   - 下采样（默认 `downscale=6`）
   - 最近邻放大回原尺寸
   - 轻微高斯模糊（`blur_radius=0.8`）
   - JPEG 压缩（`jpeg_quality=28`）
4. **Mask 生成** (`_box_mask`)：以人脸框为中心向外 padding，默认 padding 比例为 `0.35`，白色表示编辑区域。

### 3.3 本地人脸图像的保 ID 合成编辑

当只有一批包含人脸的普通图片、没有真实编辑前后对时，可使用 `face_aug_preserve`。它会先检测人脸框并过滤无脸图，再按 `--synthetic_edit_mix` 采样三类任务：

| 任务 | 默认占比 | source / target 构造 | 训练意图 |
|------|----------|----------------------|----------|
| `restore` | `0.4` | source 为小脸退化图，target 为原图 | 学小脸细节复原。 |
| `background` | `0.4` | source 为原图，target 为非脸区域颜色/曝光/模糊/噪声等合成编辑，脸区域 paste 回原图 | 学“编辑非脸区域时保持人脸身份”。 |
| `noop` | `0.2` | source=target=原图 | 学到保 ID trigger 不应主动改脸。 |

示例：

```bash
python scripts/prepare_face_edit_data.py \
  --source face_aug_preserve \
  --image_dir /path/to/face_images \
  --output_dir data/face_edit/face_aug_train \
  --resolution 512 \
  --synthetic_edit_mix restore:0.4,background:0.4,noop:0.2 \
  --face_prompt_tag "[preserve face id]"
```

也可以直接训练：

```bash
python scripts/train_face_edit_sft_flux2.py \
  --dataset_source face_aug_preserve \
  --image_dir /path/to/face_images \
  --synthetic_edit_mix restore:0.4,background:0.4,noop:0.2 \
  --face_prompt_tag "[preserve face id]"
```

### 3.4 MagicBrush 通用编辑数据

MagicBrush 是真实图像编辑数据集，包含：
- `source_img`：编辑前的原始图像
- `target_img`：编辑后的目标图像
- `instruction`：文本编辑指令
- `mask_img`：编辑区域的掩码

无需合成退化，直接按统一格式读取即可。

### 3.5 canonical JSONL 统一格式

物化后的每条样本是一个 JSON 对象，字段如下：

```json
{
  "source_image": "images/0000000_source.jpg",
  "target_image": "images/0000000_target.jpg",
  "mask_image": "masks/0000000_mask.png",
  "instruction": "restore the small low-resolution face while preserving the rest of the image",
  "metadata": {
    "dataset": "wider_face",
    "boxes_xywh": [[x, y, w, h]],
    "synthetic_task": "restore_lowres_small_face"
  }
}
```

字段说明：

| 字段 | 类型 | 说明 |
|------|------|------|
| `source_image` | string | 相对或绝对路径，指向编辑前图片。 |
| `target_image` | string | 相对或绝对路径，指向编辑后目标图片。 |
| `mask_image` | string | 相对或绝对路径，指向单通道 mask 图片。 |
| `instruction` | string | 文本编辑指令。 |
| `metadata` | object | 可选元数据，如数据集来源、人脸框坐标、合成任务类型等。 |

> **注意**：JSONL 中的相对路径是**相对于 JSONL 文件所在目录**，而非启动命令的工作目录。

### 3.6 物化数据：从在线数据集到本地 JSONL

训练脚本可以直接在线读取 Hugging Face 数据集，但团队协作时强烈建议先物化数据，固定数据版本并避免每次训练时动态合成。

#### WIDER-FACE 小脸修复数据

```bash
python scripts/prepare_face_edit_data.py \
  --source wider_face_restore \
  --split train \
  --output_dir data/face_edit/wider_train \
  --resolution 512 \
  --max_samples 2000
```

#### MagicBrush 数据

```bash
python scripts/prepare_face_edit_data.py \
  --source magicbrush \
  --split train \
  --output_dir data/face_edit/magicbrush_train \
  --resolution 512 \
  --max_samples 2000
```

#### 物化后的目录结构

```text
data/face_edit/wider_train/
  schema.json          # 字段说明与版本
  train.jsonl          # 样本列表，每行一个 JSON 对象
  images/
    0000000_source.jpg
    0000000_target.jpg
  masks/
    0000000_mask.png
```

#### 训练时使用物化数据

```bash
--dataset_source canonical_jsonl \
--jsonl_path data/face_edit/wider_train/train.jsonl
```

### 3.7 快速数据检查

建议先用小样本做端到端检查：

```bash
python scripts/prepare_face_edit_data.py \
  --source wider_face_restore \
  --split train \
  --output_dir data/face_edit/debug_wider \
  --resolution 512 \
  --max_samples 16
```

检查重点：

- `*_source.jpg` 的小脸区域应明显退化（模糊、块状、色彩压缩）。
- `*_target.jpg` 是同一张原图的清晰版本。
- `*_mask.png` 白色区域覆盖被退化的人脸，黑色区域是背景保持区域。
- JSONL 中图片路径相对 `train.jsonl` 所在目录可解析。

---

## 4. Flux2 训练路径

Flux2 默认模型名为 `flux.2-klein-4b`，对应工具常量定义在 `flow_grpo/flux2_train_utils.py`。

所有 Flux2 脚本默认**单进程**启动；多卡并行目前没有额外封装，建议先用小 batch 验证。

### 4.1 Stage 0：T2I SFT（冒烟测试）

**入口脚本**：`scripts/train_face_t2i_sft_flux2.py`

**目的**：只用 `target_images` 做标准文本到图像 flow-matching LoRA 训练，验证 Flux2 组件、数据读取、VAE / text encoder / token path 是否全部跑通。

#### 直接读 WIDER-FACE

```bash
python scripts/train_face_t2i_sft_flux2.py \
  --dataset_source wider_face_restore \
  --output_dir logs/face_t2i/sft_flux2 \
  --resolution 512 \
  --batch_size 1 \
  --gradient_accumulation_steps 4 \
  --max_samples 2000
```

#### 使用物化数据

```bash
python scripts/train_face_t2i_sft_flux2.py \
  --dataset_source canonical_jsonl \
  --jsonl_path data/face_edit/wider_train/train.jsonl \
  --output_dir logs/face_t2i/sft_flux2 \
  --batch_size 1 \
  --gradient_accumulation_steps 4
```

#### 输出路径

```text
logs/face_t2i/sft_flux2/final/lora/
```

### 4.2 Stage 1：Edit SFT（图像编辑监督微调）

**入口脚本**：`scripts/train_face_edit_sft_flux2.py`

**目的**：训练真正的图像编辑路径。脚本会编码 source / target 图像，target latent 加噪得到 `current tokens`，source latent 变成 `reference tokens`，然后 transformer 输入为：

```text
concat(noisy_target_tokens, source_image_tokens)
```

- 只监督 **noisy target tokens** 对应的 flow target。
- mask 会被下采样到 latent / token 空间，对编辑区域加权。
- 非编辑区域仍保留最低权重，避免模型完全忽略背景。

#### 推荐启动命令

```bash
python scripts/train_face_edit_sft_flux2.py \
  --dataset_source canonical_jsonl \
  --jsonl_path data/face_edit/wider_train/train.jsonl \
  --output_dir logs/face_edit/sft_flux2 \
  --resolution 512 \
  --batch_size 1 \
  --gradient_accumulation_steps 4 \
  --learning_rate 1e-4 \
  --source_mix 0.25 \
  --save_steps 500
```

#### 从 T2I SFT LoRA 继续 warm start

```bash
python scripts/train_face_edit_sft_flux2.py \
  --dataset_source canonical_jsonl \
  --jsonl_path data/face_edit/wider_train/train.jsonl \
  --resume_lora logs/face_t2i/sft_flux2/final/lora \
  --output_dir logs/face_edit/sft_flux2_from_t2i \
  --batch_size 1 \
  --gradient_accumulation_steps 4
```

#### 关键参数说明

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `--source_mix` | `0.25` | source image tokens 与 target tokens 的混合比例，控制编辑条件强度。 |
| `--resume_lora` | `None` | 从已有的 LoRA 路径恢复训练。 |
| `--learning_rate` | `1e-4` | LoRA 参数学习率。 |
| `--guidance` | `1.0` | CFG  Guidance scale。 |

#### 输出路径

```text
logs/face_edit/sft_flux2/final/lora/
```

### 4.3 Stage 2：Edit RL / DiffusionNFT（强化学习优化）

**入口脚本**：`scripts/train_face_edit_nft_flux2.py`

**目的**：从 Edit SFT LoRA 出发，用**旧策略**（old policy）采样多个候选编辑图，对候选图做人脸奖励打分，然后把同一条样本内的 reward 标准化为 advantage，最后用 DiffusionNFT 风格的 **forward-process loss** 继续优化 LoRA。

#### 启动命令

```bash
python scripts/train_face_edit_nft_flux2.py \
  --dataset_source canonical_jsonl \
  --jsonl_path data/face_edit/wider_train/train.jsonl \
  --sft_lora logs/face_edit/sft_flux2/final/lora \
  --output_dir logs/face_edit/nft_flux2 \
  --resolution 512 \
  --batch_size 1 \
  --num_candidates 8 \
  --num_inference_steps 4 \
  --learning_rate 5e-5 \
  --source_mix 0.25 \
  --nft_beta 0.25 \
  --kl_beta 1e-4 \
  --preserve_beta 0.05 \
  --save_steps 200
```

#### 关键参数说明

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `--sft_lora` | **必填** | Edit SFT 阶段输出的 LoRA 路径，RL 的起点。 |
| `--num_candidates` | `8` | 每个 source 采样多少个候选图，reward 在同组内做标准化。 |
| `--num_inference_steps` | `4` | Flux2 采样步数，越大越慢、质量通常越高。 |
| `--nft_beta` | `0.25` | DiffusionNFT 正/负隐式目标的插值强度，控制 reward 信号对前向过程的渗透程度。 |
| `--kl_beta` | `1e-4` | 对无 LoRA base prediction 的 KL / 距离约束权重，防止策略崩溃。 |
| `--preserve_beta` | `0.05` | 非编辑区域保持损失权重，约束背景不漂移。 |
| `--old_decay` | `0.5` | old policy adapter 的 EMA 更新比例。 |
| `--adv_clip_max` | `5.0` | advantage 裁剪上限。 |

#### 输出路径

```text
logs/face_edit/nft_flux2/final/lora/
```

### 4.4 Flux2 采样验证

训练完成后，可用以下命令做推理验证。

#### T2I 采样

```bash
python scripts/sample_flux2_t2i_edit.py \
  --lora_path logs/face_t2i/sft_flux2/final/lora \
  --prompt "a realistic street photo with a small clear human face in the background" \
  --output output/flux2_t2i.png \
  --num_steps 4 \
  --guidance 1.0
```

#### Edit 采样

```bash
python scripts/sample_flux2_t2i_edit.py \
  --lora_path logs/face_edit/nft_flux2/final/lora \
  --source_image data/face_edit/wider_train/images/0000000_source.jpg \
  --prompt "restore the small low-resolution face while preserving the rest of the image" \
  --output output/flux2_edit.png \
  --num_steps 4 \
  --guidance 1.0
```

---

## 5. SD3 训练路径

SD3 路径使用 `stabilityai/stable-diffusion-3.5-medium`，适合作为备选实现或对照实验。

### 5.1 SD3 Edit SFT

**入口脚本**：`scripts/train_face_edit_sft_sd3.py`

支持 DDP，可用 `torchrun` 启动：

```bash
torchrun --nproc_per_node=8 scripts/train_face_edit_sft_sd3.py \
  --dataset_source canonical_jsonl \
  --jsonl_path data/face_edit/wider_train/train.jsonl \
  --output_dir logs/face_edit/sft_sd3 \
  --resolution 512 \
  --train_batch_size 1 \
  --gradient_accumulation_steps 4 \
  --mixed_precision fp16 \
  --learning_rate 1e-4 \
  --source_mix 0.35 \
  --save_steps 500
```

#### SFT 损失组成

| 损失项 | 权重参数 | 说明 |
|--------|----------|------|
| Latent 编辑区 flow matching | `--edit_loss_weight` (`1.0`) | mask 区域内预测与 target 的 flow matching。 |
| 背景 preserve loss | `--preserve_loss_weight` (`0.25`) | 非编辑区预测 x0 贴近 source latent。 |
| Image-space robust loss | `--image_loss_weight` (`0.05`) | 来自 `flow_grpo.face_edit_losses.robust_edit_supervision_loss()`，对像素级对齐鲁棒。 |

#### 输出路径

```text
logs/face_edit/sft_sd3/final/lora/
```

### 5.2 SD3 Edit RL / DiffusionNFT

**入口脚本**：`scripts/train_face_edit_nft_sd3.py`

> **注意**：当前脚本显式要求单进程，`world_size > 1` 会直接报错。不要用多卡 `torchrun --nproc_per_node>1` 启动。

```bash
python scripts/train_face_edit_nft_sd3.py \
  --dataset_source canonical_jsonl \
  --jsonl_path data/face_edit/wider_train/train.jsonl \
  --sft_lora logs/face_edit/sft_sd3/final/lora \
  --output_dir logs/face_edit/nft_sd3 \
  --resolution 512 \
  --batch_size 1 \
  --num_candidates 8 \
  --num_inference_steps 25 \
  --strength 0.55 \
  --guidance_scale 4.5 \
  --mixed_precision fp16 \
  --learning_rate 5e-5
```

#### 额外参数说明

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `--strength` | `0.55` | SD3 Img2Img 的强度，越大越偏离 source。 |
| `--guidance_scale` | `4.5` | SD3 的 classifier-free guidance scale。 |

#### 输出路径

```text
logs/face_edit/nft_sd3/final/lora/
```

---

## 6. 人脸奖励（Reward）与损失（Loss）详解

### 6.1 SFT 阶段损失

SFT 阶段的核心是让模型学会“按 source + instruction 生成 target”。

#### Flux2 Edit SFT 损失

- **Flow Matching Loss**：在 latent/token 空间，只对 noisy target tokens 计算 flow matching。
- **Mask 加权**：mask 下采样到 token 空间后，对编辑区域赋予更高权重。
- **背景保留**：非编辑区域仍保留最低权重，避免模型完全丢弃背景信息。

#### SD3 Edit SFT 损失（更复杂）

SD3 SFT 使用 `flow_grpo/face_edit_losses.py` 中的 `robust_edit_supervision_loss`：

```python
total_loss = edit_weight * edit_loss + preserve_weight * preserve_loss + global_weight * global_loss
```

- `edit_loss`：mask 区域内多尺度 Charbonnier 损失（对轻微错位更鲁棒）。
- `preserve_loss`：非编辑区与 source 的多尺度 Charbonnier 损失。 
- `global_loss`：全局 Charbonnier 损失，提供整体结构约束。

### 6.2 RL 阶段奖励设计

人脸 RL 奖励在 `flow_grpo/face_edit_losses.py` 的 `FaceEditRewardScorer` 中实现。

#### 奖励项与权重

| 奖励项 | 默认权重 | 含义 | 依赖 |
|--------|----------|------|------|
| `face_detection` | `1.0` | OpenCV Haar 检测生成图裁剪区域内是否有人脸。 | `opencv-python` |
| `identity` | `1.0` | 有 `facenet-pytorch` 时用 `InceptionResnetV1` 提取 embedding 算 cosine；否则用颜色直方图 cosine 作为轻量 fallback。 | `facenet-pytorch`（可选） |
| `landmark` | `0.5` | 有 `mediapipe` 时比较 FaceMesh landmark 距离；否则比较低频灰度结构相似度。 | `mediapipe`（可选） |
| `preserve` | `0.75` | 非 mask 区域与 source 的 L1 保持分数：`preserve = clamp(1 - L1 * 4, 0, 1)`。 | 无 |
| `target` | `0.5` | mask 裁剪区域与 target 的 L1 相似度：`target_score = clamp(1 - L1 * 4, 0, 1)`。 | 无 |

#### 奖励聚合

单张生成图的总奖励为加权平均：

```
r = Σ(w_i * r_i) / Σ|w_i|
```

#### Advantage 计算

RL 脚本把同一原图的 `num_candidates` 个候选组成一组，做**组内标准化**（z-score）得到 advantage：

```
advantage = (reward - mean(group)) / std(group)
advantage = clamp(advantage, -adv_clip_max, adv_clip_max)
```

Advantage 再映射到 `[0, 1]` 的 `r`，用于 DiffusionNFT 正/负隐式目标。

#### 可选依赖安装

```bash
pip install facenet-pytorch mediapipe
```

没有这些依赖时训练仍可运行，但 `identity` / `landmark` 奖励会退化成轻量代理指标，信号会弱一些。

### 6.3 DiffusionNFT 前向过程损失

DiffusionNFT 的核心思想是：不在采样轨迹上做反向传播，而是把生成的 clean 图像重新加噪，在前向过程中用 reward 区分“正样本”和“负样本”：

```
L_NFT = r * L_positive + (1 - r) * L_negative + β_kl * L_kl + β_preserve * L_preserve
```

- `L_positive`：对高 reward 样本，鼓励模型向其前向过程靠拢。
- `L_negative`：对低 reward 样本，鼓励模型远离其前向过程。
- `L_kl`：约束当前策略与 base model（无 LoRA）的预测距离，防止崩溃。
- `L_preserve`：约束非编辑区域与 source 的一致性。

参数 `--nft_beta` 控制正/负目标的插值强度，`--kl_beta` 和 `--preserve_beta` 分别控制 KL 约束与背景保持。

---

## 7. LoRA 输出路径与恢复训练

### 路径约定

所有脚本**只训练 LoRA**，基础模型权重不保存。统一路径结构如下：

```text
logs/face_t2i/sft_flux2/checkpoint-STEP/lora/
logs/face_t2i/sft_flux2/final/lora/

logs/face_edit/sft_flux2/checkpoint-STEP/lora/
logs/face_edit/sft_flux2/final/lora/

logs/face_edit/nft_flux2/checkpoint-STEP/lora/
logs/face_edit/nft_flux2/final/lora/

logs/face_edit/sft_sd3/checkpoint-STEP/lora/
logs/face_edit/sft_sd3/final/lora/

logs/face_edit/nft_sd3/checkpoint-STEP/lora/
logs/face_edit/nft_sd3/final/lora/
```

### 恢复或串接训练

| 场景 | 参数 | 说明 |
|------|------|------|
| Flux2 Edit SFT 恢复 | `--resume_lora PATH` | 载入已有 LoRA 继续 SFT。 |
| Flux2 Edit SFT 从 T2I 启动 | `--resume_lora PATH` | 用 T2I SFT 的 LoRA 做 warm start。 |
| Flux2/SD3 Edit RL | `--sft_lora PATH` | **必填**，指向 Edit SFT 输出。RL 必须从 SFT 起点开始。 |
| SD3 Edit SFT 恢复 | `--resume_lora PATH` | 同 Flux2，支持从 checkpoint 恢复。 |

> **注意**：RL 脚本不支持 `--resume_lora`，它只支持 `--sft_lora` 指定起点。RL 过程中的中间 checkpoint 目前只能手动保存、手动恢复（加载 checkpoint 目录作为 `--sft_lora` 重新启动）。

---

## 8. 主要代码地图

### 8.1 入口脚本层

| 文件 | 作用 |
|------|------|
| `scripts/prepare_face_edit_data.py` | 数据物化入口。将 WIDER-FACE 或 MagicBrush 处理成 canonical JSONL，保存 source / target / mask 图片。 |
| `scripts/train_face_t2i_sft_flux2.py` | **Flux2 Stage 0**。T2I SFT smoke test，只用 target image 训练 LoRA，验证基础链路。 |
| `scripts/train_face_edit_sft_flux2.py` | **Flux2 Stage 1**。图像编辑 SFT，source tokens 作为参考条件，只监督 target tokens。 |
| `scripts/train_face_edit_nft_flux2.py` | **Flux2 Stage 2**。人脸编辑 RL，从 SFT LoRA 出发采样候选、打 reward、做 DiffusionNFT 更新。 |
| `scripts/sample_flux2_t2i_edit.py` | Flux2 LoRA 推理脚本，既支持 T2I，也支持带 source image 的 edit。 |
| `scripts/train_face_edit_sft_sd3.py` | **SD3 Stage 1**。SD3 图像编辑 SFT，支持 DDP，使用 mask-aware latent / image loss。 |
| `scripts/train_face_edit_nft_sd3.py` | **SD3 Stage 2**。SD3 人脸编辑 RL，当前单进程，使用 SD3 Img2Img pipeline 采样候选。 |
| `scripts/train_nft_sd3.py` | 原始 DiffusionNFT SD3 文生图 RL 主入口，面向 geneval / ocr / pickscore 等 prompt 数据，**不是人脸编辑主线**。 |
| `scripts/evaluation.py` | 原始 SD3 LoRA 文生图评估入口，支持 geneval / ocr / pickscore / drawbench。 |

### 8.2 人脸编辑核心模块层

| 文件 | 作用 |
|------|------|
| `flow_grpo/editing_data.py` | **统一编辑数据集**。处理 WIDER-FACE 合成退化、MagicBrush、canonical JSONL；输出 tensor batch。核心类：`FaceEditDataset`、`EditSample`。 |
| `flow_grpo/face_edit_losses.py` | **人脸编辑损失与奖励**。包含：<br>- `robust_edit_supervision_loss`：SD3 SFT 的 mask-aware 鲁棒损失。<br>- `FaceEditRewardScorer`：RL 阶段的人脸检测、身份、landmark、保持、目标相似度奖励。<br>- `OptionalFaceEmbedder` / `OptionalLandmarkScorer`：可选的人脸 embedding 与 landmark 打分器，缺失时自动降级。 |
| `flow_grpo/flux2_train_utils.py` | **Flux2 训练工具**。包含：<br>- `load_flux2_components`：加载 transformer / VAE / text encoder。<br>- `add_flux2_lora`：LoRA 注入（target modules: `img_in`, `txt_in`, `linear1`, `linear2`）。<br>- `encode_flux2_prompts` / `encode_flux2_images`：文本与图像编码。<br>- `flux2_image_tokens` / `flux2_ref_tokens` / `make_flux2_noisy_tokens`：token 化与加噪。<br>- `flux2_predict` / `flux2_sample`：训练前向与采样。<br>- `flux2_checkpoint_dir`：checkpoint 路径管理。 |
| `flow_grpo/sd3_edit_train_utils.py` | **SD3 训练工具**。包含 LoRA 注入、prompt embedding、VAE encode/decode、flow timestep、edit noisy latent 工具等。 |

### 8.3 原始 DiffusionNFT / Flow-GRPO 模块层

| 文件/目录 | 作用 |
|-----------|------|
| `config/base.py` | 原始训练配置基类。 |
| `config/nft.py` | SD3 原始 NFT 训练配置，定义 geneval / ocr / pickscore / hpsv2 / multi_reward 等奖励组合。 |
| `flow_grpo/rewards.py` | 原始 T2I reward 聚合入口。 |
| `flow_grpo/stat_tracking.py` | prompt 级 reward / advantage 统计跟踪。 |
| `flow_grpo/diffusers_patch/pipeline_with_logprob.py` | 原始采样和 logprob patch。 |
| `flow_grpo/diffusers_patch/solver.py` | 原始采样 solver。 |
| `flow_grpo/diffusers_patch/train_dreambooth_lora_sd3.py` | SD3 prompt encoding / LoRA 训练借用代码。 |
| `flow_grpo/*_scorer.py` | PickScore、ImageReward、Aesthetic、CLIP、HPSv2、UnifiedReward 等 reward scorer。 |
| `dataset/` | 原始 T2I prompt 数据，如 geneval、ocr、pickscore、drawbench。 |

### 模块依赖关系

```
入口脚本 (scripts/train_*.py)
    ├── flow_grpo/editing_data.py          # 数据加载
    ├── flow_grpo/face_edit_losses.py      # SFT loss + RL reward
    ├── flow_grpo/flux2_train_utils.py     # Flux2 模型/LoRA/采样工具
    └── flow_grpo/sd3_edit_train_utils.py  # SD3 模型/LoRA/采样工具
```

---

## 9. 常见问题与排查

| 问题 | 原因 | 解决 |
|------|------|------|
| `FaceEditDataset` 报错 `jsonl_path is required` | 使用 `canonical_jsonl` 时未传 `--jsonl_path`。 | 检查是否传入 `--jsonl_path data/.../train.jsonl`。 |
| 图片路径解析失败 | JSONL 中的相对路径解析基准错误。 | JSONL 中的相对路径是**相对于 JSONL 文件所在目录**，不是启动目录。 |
| `ImportError: Flux2 training requires...` | 未安装官方 `flux2` Python 包。 | 执行 `pip install git+https://github.com/black-forest-labs/flux2.git` 或设置 `PYTHONPATH`。 |
| RL 训练 OOM | `num_candidates` 太大，显存随候选数线性增长。 | 先用 `--max_samples 16 --num_candidates 2` 做冒烟测试，再逐步放大。 |
| SD3 RL 报错 `world_size > 1 not supported` | 使用了多卡 `torchrun` 启动 SD3 RL。 | SD3 RL 目前**只能单进程**跑；SD3 SFT 可以多卡。 |
| identity/landmark 奖励始终很低 | 未安装 `facenet-pytorch` / `mediapipe`。 | 安装 `pip install facenet-pytorch mediapipe`，否则奖励会降级为颜色直方图/灰度结构相似度。 |
| WIDER-FACE 合成数据缺乏身份保持信号 | 合成退化只保证了“低清→高清”，不等价于真实身份保持数据。 | 扩展到更泛化的人脸编辑时，建议混入 MagicBrush 或自有 canonical JSONL。 |

---

*文档版本：2026-04-28*  
*如有问题，请优先检查入口脚本的 `parse_args()` 与本文档的参数表格。*
