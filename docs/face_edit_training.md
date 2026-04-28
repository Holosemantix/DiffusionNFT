# Small-Face Edit Training

This branch adds a two-stage training path for improving small degraded faces. The default base model is now
`FLUX.2 [klein] 4B` (`flux.2-klein-4b` / `black-forest-labs/FLUX.2-klein-4B`) instead of SD3.5.

The validation order is:

1. T2I SFT: validate the Flux2 flow/token training path without image conditioning.
2. Edit SFT: validate Flux2's edit path, where noisy target image tokens are concatenated with source/edit image tokens.
3. Edit RL: start from the edit SFT LoRA and apply DiffusionNFT-style online RL using face-aware rewards.

The initial task is intentionally narrow: restore small low-resolution faces while preserving the rest of the image.
This keeps the reward signal focused before expanding to general multi-edit instructions.

## Dataset Sources

- `wider_face_restore`: downloads `CUHK-CSE/wider_face` through Hugging Face Datasets and constructs pairs by degrading annotated small face boxes. The target is the original image.
- `magicbrush`: downloads `osunlp/MagicBrush` through Hugging Face Datasets and reads real `source_img, instruction, target_img, mask_img` edit triples.
- `canonical_jsonl`: reads materialized samples with `source_image`, `target_image`, `mask_image`, `instruction`, and `metadata`.

To materialize a cacheable JSONL dataset:

```bash
python scripts/prepare_face_edit_data.py \
  --source wider_face_restore \
  --split train \
  --output_dir data/face_edit/wider_train \
  --max_samples 2000
```

## SFT

Install the optional Flux2 dependency:

```bash
pip install -e ".[flux2,face]"
```

T2I path:

```bash
python scripts/train_face_t2i_sft_flux2.py \
  --dataset_source wider_face_restore \
  --output_dir logs/face_t2i/sft_flux2 \
  --batch_size 1
```

Edit path:

```bash
python scripts/train_face_edit_sft_flux2.py \
  --dataset_source wider_face_restore \
  --output_dir logs/face_edit/sft_flux2 \
  --batch_size 1 \
  --gradient_accumulation_steps 4
```

The edit SFT objective uses Flux2 LoRA and trains a flow-matching target on target image tokens.
For editing, the transformer receives `concat(noisy_target_tokens, source_image_tokens)` and only the noisy target tokens are supervised.

## RL

```bash
python scripts/train_face_edit_nft_flux2.py \
  --dataset_source wider_face_restore \
  --sft_lora logs/face_edit/sft_flux2/final/lora \
  --output_dir logs/face_edit/nft_flux2 \
  --num_candidates 8
```

The RL reward includes face detection, identity similarity, landmark/structure similarity, target similarity, and non-edit-region preservation. Optional `facenet-pytorch` and `mediapipe` improve identity and landmark scoring; deterministic fallbacks keep the pipeline runnable without them.

## Sampling

T2I:

```bash
python scripts/sample_flux2_t2i_edit.py \
  --lora_path logs/face_t2i/sft_flux2/final/lora \
  --prompt "a realistic street photo with a small clear human face in the background" \
  --output output/flux2_t2i.png
```

Edit:

```bash
python scripts/sample_flux2_t2i_edit.py \
  --lora_path logs/face_edit/nft_flux2/final/lora \
  --source_image path/to/source.jpg \
  --prompt "restore the small low-resolution face while preserving the rest of the image" \
  --output output/flux2_edit.png
```
