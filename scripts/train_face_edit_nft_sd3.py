# SPDX-License-Identifier: Apache-2.0
"""DiffusionNFT-style RL stage for small-face image editing.

This script starts from an SFT LoRA checkpoint, samples multiple edited
candidates per source image with SD3 img2img, scores them with face/edit
rewards, and applies the forward-process NFT loss to the generated clean
latents.
"""

import argparse
import os

import torch
import torch.nn.functional as F
from diffusers import StableDiffusion3Img2ImgPipeline
from peft import PeftModel
from torch.utils.data import DataLoader
from tqdm import tqdm
from torchvision.transforms.functional import to_pil_image

from flow_grpo.device_utils import get_device, get_grad_scaler, get_amp_context, get_pin_memory
from flow_grpo.editing_data import FaceEditDataset, collate_edit_samples
from flow_grpo.face_edit_losses import FaceEditRewardScorer
from flow_grpo.sd3_edit_train_utils import (
    SD3_MODEL_ID,
    add_sd3_lora,
    compute_prompt_embeddings,
    encode_vae_images,
    is_main_process,
    make_edit_flow_noisy_latents,
    sample_flow_timestep,
    setup_distributed,
)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset_source", default="wider_face_restore", choices=["wider_face_restore", "magicbrush", "canonical_jsonl"])
    parser.add_argument("--dataset_split", default="train")
    parser.add_argument("--jsonl_path", default=None)
    parser.add_argument("--cache_dir", default=None)
    parser.add_argument("--pretrained_model", default=SD3_MODEL_ID)
    parser.add_argument("--sft_lora", required=True)
    parser.add_argument("--output_dir", default="logs/face_edit/nft_sd3")
    parser.add_argument("--resolution", type=int, default=512)
    parser.add_argument("--max_samples", type=int, default=None)
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--num_candidates", type=int, default=8)
    parser.add_argument("--num_epochs", type=int, default=1)
    parser.add_argument("--num_inference_steps", type=int, default=25)
    parser.add_argument("--strength", type=float, default=0.55)
    parser.add_argument("--guidance_scale", type=float, default=4.5)
    parser.add_argument("--learning_rate", type=float, default=5e-5)
    parser.add_argument("--mixed_precision", default="fp16", choices=["fp16", "bf16", "no"])
    parser.add_argument("--source_mix", type=float, default=0.35)
    parser.add_argument("--nft_beta", type=float, default=0.25)
    parser.add_argument("--kl_beta", type=float, default=1e-4)
    parser.add_argument("--preserve_beta", type=float, default=0.05)
    parser.add_argument("--adv_clip_max", type=float, default=5.0)
    parser.add_argument("--old_decay", type=float, default=0.5)
    parser.add_argument("--save_steps", type=int, default=200)
    parser.add_argument("--num_workers", type=int, default=2)
    return parser.parse_args()


def _repeat_batch(batch, repeats: int, device):
    return {
        "source_images": batch["source_images"].repeat_interleave(repeats, dim=0).to(device),
        "target_images": batch["target_images"].repeat_interleave(repeats, dim=0).to(device),
        "masks": batch["masks"].repeat_interleave(repeats, dim=0).to(device),
        "instructions": [p for p in batch["instructions"] for _ in range(repeats)],
        "metadata": [m for m in batch["metadata"] for _ in range(repeats)],
    }


def _group_advantages(rewards: torch.Tensor, batch_size: int, num_candidates: int, adv_clip_max: float):
    rewards_grouped = rewards.view(batch_size, num_candidates)
    mean = rewards_grouped.mean(dim=1, keepdim=True)
    std = rewards_grouped.std(dim=1, keepdim=True).clamp_min(1e-4)
    advantages = ((rewards_grouped - mean) / std).clamp(-adv_clip_max, adv_clip_max)
    return advantages.reshape(-1)


def main():
    args = parse_args()
    rank, world_size, local_rank = setup_distributed()
    if world_size > 1:
        raise RuntimeError("train_face_edit_nft_sd3.py currently expects one process; use batch_size/num_candidates for grouping.")
    device = get_device(local_rank)
    dtype = {"fp16": torch.float16, "bf16": torch.bfloat16, "no": torch.float32}[args.mixed_precision]
    amp_enabled = args.mixed_precision != "no"

    if is_main_process(rank):
        os.makedirs(args.output_dir, exist_ok=True)

    dataset = FaceEditDataset(
        source=args.dataset_source,
        split=args.dataset_split,
        resolution=args.resolution,
        cache_dir=args.cache_dir,
        jsonl_path=args.jsonl_path,
        max_samples=args.max_samples,
    )
    dataloader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        collate_fn=collate_edit_samples,
        pin_memory=get_pin_memory(device),
    )

    pipeline = StableDiffusion3Img2ImgPipeline.from_pretrained(args.pretrained_model)
    pipeline.vae.requires_grad_(False)
    pipeline.text_encoder.requires_grad_(False)
    pipeline.text_encoder_2.requires_grad_(False)
    pipeline.text_encoder_3.requires_grad_(False)
    pipeline.transformer.requires_grad_(False)
    pipeline.safety_checker = None

    pipeline.vae.to(device, dtype=torch.float32)
    text_dtype = dtype if amp_enabled else torch.float32
    pipeline.text_encoder.to(device, dtype=text_dtype)
    pipeline.text_encoder_2.to(device, dtype=text_dtype)
    pipeline.text_encoder_3.to(device, dtype=text_dtype)
    transformer, lora_config = add_sd3_lora(pipeline.transformer.to(device), args.sft_lora, adapter_name="default")
    transformer.add_adapter("old", lora_config)
    transformer.set_adapter("default")

    default_params = [p for p in transformer.parameters() if p.requires_grad]
    transformer.set_adapter("old")
    old_params = [p for p in transformer.parameters() if p.requires_grad]
    transformer.set_adapter("default")
    with torch.no_grad():
        for src, tgt in zip(default_params, old_params, strict=True):
            tgt.copy_(src.detach())

    pipeline.transformer = transformer
    optimizer = torch.optim.AdamW(default_params, lr=args.learning_rate, weight_decay=1e-4)
    scaler = get_grad_scaler(device.type, enabled=args.mixed_precision == "fp16")
    reward_scorer = FaceEditRewardScorer(device)
    text_encoders = [pipeline.text_encoder, pipeline.text_encoder_2, pipeline.text_encoder_3]
    tokenizers = [pipeline.tokenizer, pipeline.tokenizer_2, pipeline.tokenizer_3]

    global_step = 0
    for epoch in range(args.num_epochs):
        iterator = tqdm(dataloader, disable=not is_main_process(rank), desc=f"NFT edit epoch {epoch}")
        for raw_batch in iterator:
            batch = _repeat_batch(raw_batch, args.num_candidates, device)
            source_pil = [to_pil_image(img.cpu().clamp(0, 1)) for img in batch["source_images"]]

            transformer.set_adapter("old")
            with torch.no_grad(), get_amp_context(device.type, enabled=amp_enabled, dtype=dtype):
                generated = pipeline(
                    prompt=batch["instructions"],
                    image=source_pil,
                    strength=args.strength,
                    num_inference_steps=args.num_inference_steps,
                    guidance_scale=args.guidance_scale,
                    output_type="pt",
                    height=args.resolution,
                    width=args.resolution,
                )[0]
            transformer.set_adapter("default")

            rewards, reward_details = reward_scorer.score_batch(
                generated=generated,
                source=batch["source_images"],
                target=batch["target_images"],
                masks=batch["masks"],
            )
            advantages = _group_advantages(rewards, raw_batch["source_images"].shape[0], args.num_candidates, args.adv_clip_max)

            with torch.no_grad():
                prompt_embeds, pooled_embeds = compute_prompt_embeddings(batch["instructions"], text_encoders, tokenizers, 128, device)
                source_latents = encode_vae_images(pipeline.vae, batch["source_images"], torch.float32)
                generated_latents = encode_vae_images(pipeline.vae, generated.to(device), torch.float32)

            t = sample_flow_timestep(generated_latents.shape[0], device)
            xt, _ = make_edit_flow_noisy_latents(generated_latents, source_latents, t, source_mix=args.source_mix)
            timesteps = (t * 1000).long()
            t_expanded = t.view(-1, *([1] * (generated_latents.ndim - 1)))

            with get_amp_context(device.type, enabled=amp_enabled, dtype=dtype):
                transformer.set_adapter("old")
                with torch.no_grad():
                    old_prediction = transformer(
                        hidden_states=xt.to(dtype),
                        timestep=timesteps,
                        encoder_hidden_states=prompt_embeds,
                        pooled_projections=pooled_embeds,
                        return_dict=False,
                    )[0].detach().float()

                transformer.set_adapter("default")
                forward_prediction = transformer(
                    hidden_states=xt.to(dtype),
                    timestep=timesteps,
                    encoder_hidden_states=prompt_embeds,
                    pooled_projections=pooled_embeds,
                    return_dict=False,
                )[0].float()

                with torch.no_grad():
                    with transformer.disable_adapter():
                        ref_prediction = transformer(
                            hidden_states=xt.to(dtype),
                            timestep=timesteps,
                            encoder_hidden_states=prompt_embeds,
                            pooled_projections=pooled_embeds,
                            return_dict=False,
                        )[0].float()

                normalized_adv = (advantages / args.adv_clip_max) / 2.0 + 0.5
                r = normalized_adv.clamp(0, 1).view(-1, *([1] * (generated_latents.ndim - 1)))
                positive_prediction = args.nft_beta * forward_prediction + (1.0 - args.nft_beta) * old_prediction
                implicit_negative_prediction = (1.0 + args.nft_beta) * old_prediction - args.nft_beta * forward_prediction

                positive_x0 = xt.float() - t_expanded * positive_prediction
                negative_x0 = xt.float() - t_expanded * implicit_negative_prediction
                positive_loss = (positive_x0 - generated_latents.float()).pow(2).mean(dim=tuple(range(1, generated_latents.ndim)))
                negative_loss = (negative_x0 - generated_latents.float()).pow(2).mean(dim=tuple(range(1, generated_latents.ndim)))
                policy_loss = (r.flatten() * positive_loss / args.nft_beta + (1 - r.flatten()) * negative_loss / args.nft_beta).mean()

                kl_loss = (forward_prediction - ref_prediction).pow(2).mean()
                latent_mask = F.interpolate(batch["masks"].float(), size=generated_latents.shape[-2:], mode="nearest").expand_as(generated_latents)
                pred_x0 = xt.float() - t_expanded * forward_prediction
                preserve_loss = F.smooth_l1_loss(pred_x0 * (1 - latent_mask), source_latents.float() * (1 - latent_mask))
                loss = policy_loss + args.kl_beta * kl_loss + args.preserve_beta * preserve_loss

            optimizer.zero_grad()
            if args.mixed_precision == "fp16":
                scaler.scale(loss).backward()
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(default_params, 1.0)
                scaler.step(optimizer)
                scaler.update()
            else:
                loss.backward()
                torch.nn.utils.clip_grad_norm_(default_params, 1.0)
                optimizer.step()

            with torch.no_grad():
                transformer.set_adapter("old")
                for src, tgt in zip(default_params, old_params, strict=True):
                    tgt.copy_(tgt.detach() * args.old_decay + src.detach() * (1.0 - args.old_decay))
                transformer.set_adapter("default")

            if is_main_process(rank):
                logs = {
                    "loss": float(loss.detach()),
                    "policy": float(policy_loss.detach()),
                    "kl": float(kl_loss.detach()),
                    "preserve": float(preserve_loss.detach()),
                    "reward": float(rewards.mean().detach()),
                    **{f"r_{k}": float(v.mean().detach()) for k, v in reward_details.items()},
                }
                iterator.set_postfix(logs)

            global_step += 1
            if is_main_process(rank) and args.save_steps > 0 and global_step % args.save_steps == 0:
                save_dir = os.path.join(args.output_dir, f"checkpoint-{global_step}", "lora")
                os.makedirs(save_dir, exist_ok=True)
                transformer.save_pretrained(save_dir)

    if is_main_process(rank):
        save_dir = os.path.join(args.output_dir, "final", "lora")
        os.makedirs(save_dir, exist_ok=True)
        transformer.save_pretrained(save_dir)


if __name__ == "__main__":
    main()

