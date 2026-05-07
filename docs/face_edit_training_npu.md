# 人脸编辑 SFT/RL NPU 训练完整指南

> 本文档是 `docs/face_edit_training.md` 的 **NPU 适配版本**，对应分支 `v01_lyr_face_npu_dev`。  
> 所有 GPU 相关代码已自动兼容华为 Ascend NPU（`torch-npu`），单卡/多卡、Flux2/SD3 均可直接启动。

---

## 目录

- [1. NPU 适配概览](#1-npu-适配概览)
- [2. 环境准备](#2-环境准备)
- [3. 推荐运行顺序](#3-推荐运行顺序)
- [4. 数据集处理](#4-数据集处理)
- [5. Flux2 NPU 训练路径](#5-flux2-npu-训练路径)
  - [5.1 Stage 0：T2I SFT](#51-stage-0t2i-sft)
  - [5.2 Stage 1：Edit SFT](#52-stage-1edit-sft)
  - [5.3 Stage 2：Edit RL / DiffusionNFT](#53-stage-2edit-rldiffusionnft)
  - [5.4 Flux2 采样验证](#54-flux2-采样验证)
- [6. SD3 NPU 训练路径](#6-sd3-npu-训练路径)
  - [6.1 SD3 Edit SFT（支持多卡 DDP）](#61-sd3-edit-sft支持多卡-ddp)
  - [6.2 SD3 Edit RL](#62-sd3-edit-rl)
- [7. 人脸奖励与损失](#7-人脸奖励与损失)
- [8. LoRA 输出路径与恢复训练](#8-lora-输出路径与恢复训练)
- [9. 主要代码地图](#9-主要代码地图)
- [10. NPU 常见问题与排查](#10-npu-常见问题与排查)

---

## 1. NPU 适配概览

本分支基于 `v01_lyr_face_dev` 做最小化 NPU 适配，**训练逻辑、超参、数据格式全部保持不变**。

### 自动兼容机制

| 机制 | GPU (CUDA) | NPU (Ascend) |
|------|------------|--------------|
| **Device 选择** | `torch.cuda` | `torch.npu` 自动优先（见 `flow_grpo/device_utils.py`） |
| **DataLoader `pin_memory`** | `True` | 自动关闭，避免 NPU 兼容问题 |
| **随机数生成** | `torch.Generator("cuda")` | CPU Generator + `.to(npu)` fallback |
| **分布式 Backend** | `nccl` | 自动切换为 `hccl` |
| **AMP (fp16/bf16)** | `torch.cuda.amp` | 自动降级为 `torch.npu.amp` / `torch.amp` |

### 新增/修改文件

```text
flow_grpo/device_utils.py                 # 新增：get_device / get_pin_memory / get_amp_context / get_grad_scaler
flow_grpo/flux2_train_utils.py            # 修改：flux2_sample 的 generator NPU 兼容
flow_grpo/sd3_edit_train_utils.py         # 修改：setup_distributed 自动识别 hccl
scripts/train_face_edit_nft_flux2.py      # 修改：device & pin_memory
scripts/train_face_edit_sft_flux2.py      # 修改：device & pin_memory
scripts/train_face_t2i_sft_flux2.py       # 修改：device & pin_memory
scripts/sample_flux2_t2i_edit.py          # 修改：device
scripts/train_face_edit_sft_sd3.py        # 修改：device / pin_memory / amp
scripts/train_face_edit_nft_sd3.py        # 修改：device / pin_memory / amp
```

---

## 2. 环境准备

### 预装环境（已验证）

当前 NPU 环境已预装以下关键包，**无需再按 setup.py 中的 torch==2.6.0 重装**：

```text
torch               2.1.0
torch-npu           2.1.0.post8.dev20241029
torchvision         0.16.0
apex                0.1.dev20241029+ascend
hccl                0.1.0
transformers        4.45.0
accelerate          1.0.1
```

### 安装项目依赖

```bash
pip install -e .
# 或仅安装 face 可选依赖（facenet-pytorch / mediapipe 为可选）
pip install facenet-pytorch mediapipe
```

> **注意**：`setup.py` 中写的 `torch==2.6.0` 面向 CUDA 环境；NPU 上保持现有 `torch==2.1.0` + `torch-npu` 即可。

### 模型缓存

模型与数据集来自 Hugging Face，受限模型或私有缓存环境需提前执行：

```bash
huggingface-cli login
```

---

## 3. 推荐运行顺序

NPU 上推荐先准备环境和本地数据，再启动训练。不要直接在训练命令里动态加载
`wider_face_restore`，这样可以避免训练进程反复触发 Hugging Face 远程数据集脚本和缓存逻辑。

1. 登录 Hugging Face，确保模型和数据集可下载：

```bash
huggingface-cli login
```

2. 物化 WIDER FACE 编辑数据到本地 JSONL：

```bash
python scripts/prepare_face_edit_data.py \
  --source wider_face_restore \
  --split train \
  --output_dir data/face_edit/wider_train \
  --resolution 512 \
  --max_samples 2000
```

3. 先跑小样本 Edit SFT 冒烟测试：

```bash
python scripts/train_face_edit_sft_flux2.py \
  --dataset_source canonical_jsonl \
  --jsonl_path data/face_edit/wider_train/train.jsonl \
  --output_dir logs/face_edit/debug_sft_flux2_npu \
  --resolution 512 \
  --batch_size 1 \
  --max_samples 16 \
  --num_epochs 1 \
  --save_steps 10
```

4. 冒烟通过后，按目标模型路径继续训练：

- Flux2 推荐顺序：`T2I SFT -> Edit SFT -> Edit RL -> 采样验证`。
- 只做编辑修复时，可以直接从 `Edit SFT` 开始；需要 RL 提升时，再用 `Edit SFT` 的 LoRA 启动 `Edit RL`。
- SD3 推荐先跑 `SD3 Edit SFT`，再单进程跑 `SD3 Edit RL`。

---

## 4. 数据集处理

与 GPU 版本完全一致。建议先物化数据，避免每次训练动态合成：

```bash
python scripts/prepare_face_edit_data.py \
  --source wider_face_restore \
  --split train \
  --output_dir data/face_edit/wider_train \
  --resolution 512 \
  --max_samples 2000
```

训练时通过 `--dataset_source canonical_jsonl --jsonl_path data/face_edit/wider_train/train.jsonl` 使用。

如果直接训练时使用 `--dataset_source wider_face_restore` 并在 `load_dataset("CUHK-CSE/wider_face")`
阶段报 `UnicodeDecodeError: 'utf-8' codec can't decode byte 0x8b`，通常是 Hugging Face
数据集脚本/缓存里混入了 gzip 二进制文件。先清理 WIDER FACE 相关 HF 缓存，或换一个新的
`--cache_dir` 重新下载；NPU 训练更推荐先运行上面的 `prepare_face_edit_data.py`，再用
`canonical_jsonl` 读取本地物化数据，避免训练进程反复触发远程数据集加载逻辑。

---

## 5. Flux2 NPU 训练路径

Flux2 默认模型名为 `flux.2-klein-4b`。NPU 上**建议单进程启动**；多卡并行目前无额外封装，和 GPU 版本保持一致。

### 5.1 Stage 0：T2I SFT

```bash
python scripts/train_face_t2i_sft_flux2.py \
  --dataset_source canonical_jsonl \
  --jsonl_path data/face_edit/wider_train/train.jsonl \
  --output_dir logs/face_t2i/sft_flux2_npu \
  --resolution 512 \
  --batch_size 1 \
  --gradient_accumulation_steps 4
```

### 5.2 Stage 1：Edit SFT

```bash
python scripts/train_face_edit_sft_flux2.py \
  --dataset_source canonical_jsonl \
  --jsonl_path data/face_edit/wider_train/train.jsonl \
  --output_dir logs/face_edit/sft_flux2_npu \
  --resolution 512 \
  --batch_size 1 \
  --gradient_accumulation_steps 4 \
  --learning_rate 1e-4 \
  --source_mix 0.25 \
  --save_steps 500
```

从 T2I SFT LoRA 继续 warm start：

```bash
python scripts/train_face_edit_sft_flux2.py \
  --dataset_source canonical_jsonl \
  --jsonl_path data/face_edit/wider_train/train.jsonl \
  --resume_lora logs/face_t2i/sft_flux2_npu/final/lora \
  --output_dir logs/face_edit/sft_flux2_from_t2i_npu \
  --batch_size 1 \
  --gradient_accumulation_steps 4
```

### 5.3 Stage 2：Edit RL / DiffusionNFT

```bash
python scripts/train_face_edit_nft_flux2.py \
  --dataset_source canonical_jsonl \
  --jsonl_path data/face_edit/wider_train/train.jsonl \
  --sft_lora logs/face_edit/sft_flux2_npu/final/lora \
  --output_dir logs/face_edit/nft_flux2_npu \
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

### 5.4 Flux2 采样验证

#### T2I 采样

```bash
python scripts/sample_flux2_t2i_edit.py \
  --lora_path logs/face_t2i/sft_flux2_npu/final/lora \
  --prompt "a realistic street photo with a small clear human face in the background" \
  --output output/flux2_t2i_npu.png \
  --num_steps 4 \
  --guidance 1.0
```

#### Edit 采样

```bash
python scripts/sample_flux2_t2i_edit.py \
  --lora_path logs/face_edit/nft_flux2_npu/final/lora \
  --source_image data/face_edit/wider_train/images/0000000_source.jpg \
  --prompt "restore the small low-resolution face while preserving the rest of the image" \
  --output output/flux2_edit_npu.png \
  --num_steps 4 \
  --guidance 1.0
```

---

## 6. SD3 NPU 训练路径

SD3 路径使用 `stabilityai/stable-diffusion-3.5-medium`。NPU 上 DDP backend 已自动切换为 `hccl`，可直接用 `torchrun` 启动多卡。

### 6.1 SD3 Edit SFT（支持多卡 DDP）

单卡：

```bash
python scripts/train_face_edit_sft_sd3.py \
  --dataset_source canonical_jsonl \
  --jsonl_path data/face_edit/wider_train/train.jsonl \
  --output_dir logs/face_edit/sft_sd3_npu \
  --resolution 512 \
  --train_batch_size 1 \
  --gradient_accumulation_steps 4 \
  --mixed_precision fp16 \
  --learning_rate 1e-4 \
  --source_mix 0.35 \
  --save_steps 500
```

多卡（8 卡 NPU，自动使用 hccl）：

```bash
torchrun --nproc_per_node=8 scripts/train_face_edit_sft_sd3.py \
  --dataset_source canonical_jsonl \
  --jsonl_path data/face_edit/wider_train/train.jsonl \
  --output_dir logs/face_edit/sft_sd3_npu \
  --resolution 512 \
  --train_batch_size 1 \
  --gradient_accumulation_steps 4 \
  --mixed_precision fp16 \
  --learning_rate 1e-4 \
  --source_mix 0.35 \
  --save_steps 500
```

### 6.2 SD3 Edit RL

> **注意**：当前脚本显式要求单进程，`world_size > 1` 会直接报错。不要多卡启动。

```bash
python scripts/train_face_edit_nft_sd3.py \
  --dataset_source canonical_jsonl \
  --jsonl_path data/face_edit/wider_train/train.jsonl \
  --sft_lora logs/face_edit/sft_sd3_npu/final/lora \
  --output_dir logs/face_edit/nft_sd3_npu \
  --resolution 512 \
  --batch_size 1 \
  --num_candidates 8 \
  --num_inference_steps 25 \
  --strength 0.55 \
  --guidance_scale 4.5 \
  --mixed_precision fp16 \
  --learning_rate 5e-5
```

---

## 7. 人脸奖励与损失

与 GPU 版本完全一致，详见原文档 `docs/face_edit_training.md` 第 6 节。  
简要说明：

- **SFT 阶段**：mask-aware flow matching + preserve loss。
- **RL 阶段**：`FaceEditRewardScorer` 聚合 detection / identity / landmark / preserve / target 奖励，组内标准化得到 advantage，再用 DiffusionNFT 前向过程损失优化 LoRA。

可选依赖安装（提升奖励信号精度）：

```bash
pip install facenet-pytorch mediapipe
```

---

## 8. LoRA 输出路径与恢复训练

与 GPU 版本完全一致：

```text
logs/face_t2i/sft_flux2_npu/final/lora/
logs/face_edit/sft_flux2_npu/final/lora/
logs/face_edit/nft_flux2_npu/final/lora/
logs/face_edit/sft_sd3_npu/final/lora/
logs/face_edit/nft_sd3_npu/final/lora/
```

| 场景 | 参数 | 说明 |
|------|------|------|
| Flux2 Edit SFT 恢复 | `--resume_lora PATH` | 载入已有 LoRA 继续 SFT。 |
| Flux2 Edit RL | `--sft_lora PATH` | **必填**，指向 Edit SFT 输出。 |
| SD3 Edit SFT 恢复 | `--resume_lora PATH` | 同 Flux2。 |

---

## 9. 主要代码地图

### 新增 NPU 兼容层

| 文件 | 作用 |
|------|------|
| `flow_grpo/device_utils.py` | **NPU/CUDA/CPU 自动检测与适配**。包含 `get_device`、`get_pin_memory`、`get_amp_context`、`get_grad_scaler`。所有入口脚本统一调用。 |

### 入口脚本层

| 文件 | 作用 |
|------|------|
| `scripts/train_face_t2i_sft_flux2.py` | Flux2 Stage 0（NPU）。 |
| `scripts/train_face_edit_sft_flux2.py` | Flux2 Stage 1（NPU）。 |
| `scripts/train_face_edit_nft_flux2.py` | Flux2 Stage 2（NPU）。 |
| `scripts/sample_flux2_t2i_edit.py` | Flux2 采样验证（NPU）。 |
| `scripts/train_face_edit_sft_sd3.py` | SD3 Stage 1（NPU，支持 hccl 多卡）。 |
| `scripts/train_face_edit_nft_sd3.py` | SD3 Stage 2（NPU，单进程）。 |

### 核心模块层

| 文件 | 作用 |
|------|------|
| `flow_grpo/editing_data.py` | 统一编辑数据集（无需修改）。 |
| `flow_grpo/face_edit_losses.py` | 人脸编辑损失与奖励（无需修改）。 |
| `flow_grpo/flux2_train_utils.py` | Flux2 模型/LoRA/采样工具（已适配 generator）。 |
| `flow_grpo/sd3_edit_train_utils.py` | SD3 训练工具（已适配 distributed backend）。 |

---

## 10. NPU 常见问题与排查

| 问题 | 原因 | 解决 |
|------|------|------|
| `RuntimeError: NPU not available` | `torch-npu` 未正确安装或环境变量未导出。 | 执行 `npu-smi info` 确认驱动；检查 `python -c "import torch; print(torch.npu.is_available())"`。 |
| `ImportError: Flux2 training requires...` | 未安装官方 `flux2` Python 包。 | 同 GPU 版本：`pip install git+https://github.com/black-forest-labs/flux2.git` 或设置 `PYTHONPATH=/path/to/flux2/src`。 |
| DataLoader `pin_memory` 报错 | NPU 对 pin_memory 支持有限。 | 已由 `device_utils.get_pin_memory()` 自动关闭，无需手动处理。 |
| SD3 DDP 报错 `no backend nccl` | NPU 不支持 nccl。 | 已由 `setup_distributed()` 自动切换为 `hccl`，确保使用本分支代码。 |
| `GradScaler` 在 NPU 上报错 | `torch.cuda.amp.GradScaler` 不识别 npu 设备。 | 已由 `get_grad_scaler()` 自动适配为 `torch.npu.amp.GradScaler` 或 `torch.amp.GradScaler`。 |
| RL 训练 OOM | `num_candidates` 太大，NPU 显存随候选数线性增长。 | 先用 `--max_samples 16 --num_candidates 2` 做冒烟测试，再逐步放大。 |
| SD3 RL 报错 `world_size > 1 not supported` | 使用了多卡 `torchrun` 启动 SD3 RL。 | SD3 RL 目前**只能单进程**跑；SD3 SFT 可以多卡。 |
| identity/landmark 奖励始终很低 | 未安装 `facenet-pytorch` / `mediapipe`。 | 安装 `pip install facenet-pytorch mediapipe`，否则奖励会降级为颜色直方图/灰度结构相似度。 |

---

*文档版本：2026-05-05*  
*分支：`v01_lyr_face_npu_dev`*
