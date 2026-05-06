# SPDX-License-Identifier: Apache-2.0
"""Flux2 klein 4B DiffusionNFT RL for small-face editing."""

import argparse
import os

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm

from flow_grpo.device_utils import get_device, get_pin_memory
from flow_grpo.editing_data import FaceEditDataset, collate_edit_samples
from flow_grpo.face_edit_losses import FaceEditRewardScorer
from flow_grpo.flux2_train_utils import (
    FLUX2_KLEIN_4B,
    add_flux2_lora,
    encode_flux2_images,
    encode_flux2_prompts,
    flux2_checkpoint_dir,
    flux2_image_tokens,
    flux2_predict,
    flux2_ref_tokens,
    flux2_sample,
    load_flux2_components,
    make_flux2_noisy_tokens,
    sample_flux2_timestep,
)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_name", default=FLUX2_KLEIN_4B)
    parser.add_argument("--sft_lora", required=True)
    parser.add_argument("--dataset_source", default="wider_face_restore", choices=["wider_face_restore", "magicbrush", "canonical_jsonl"])
    parser.add_argument("--dataset_split", default="train")
    parser.add_argument("--jsonl_path", default=None)
    parser.add_argument("--cache_dir", default=None)
    parser.add_argument("--output_dir", default="logs/face_edit/nft_flux2")
    parser.add_argument("--resolution", type=int, default=512)
    parser.add_argument("--max_samples", type=int, default=None)
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--num_candidates", type=int, default=8)
    parser.add_argument("--num_epochs", type=int, default=1)
    parser.add_argument("--num_inference_steps", type=int, default=4)
    parser.add_argument("--guidance", type=float, default=1.0)
    parser.add_argument("--learning_rate", type=float, default=5e-5)
    parser.add_argument("--source_mix", type=float, default=0.25)
    parser.add_argument("--nft_beta", type=float, default=0.25)
    parser.add_argument("--kl_beta", type=float, default=1e-4)
    parser.add_argument("--preserve_beta", type=float, default=0.05)
    parser.add_argument("--adv_clip_max", type=float, default=5.0)
    parser.add_argument("--old_decay", type=float, default=0.5)
    parser.add_argument("--save_steps", type=int, default=200)
    parser.add_argument("--num_workers", type=int, default=2)
    parser.add_argument("--debug_model", action="store_true")
    return parser.parse_args()


def repeat_batch(batch, repeats: int, device):
    return {
        "source_images": batch["source_images"].repeat_interleave(repeats, dim=0).to(device),
        "target_images": batch["target_images"].repeat_interleave(repeats, dim=0).to(device),
        "masks": batch["masks"].repeat_interleave(repeats, dim=0).to(device),
        "instructions": [p for p in batch["instructions"] for _ in range(repeats)],
        "metadata": [m for m in batch["metadata"] for _ in range(repeats)],
    }


def group_advantages(rewards: torch.Tensor, batch_size: int, num_candidates: int, adv_clip_max: float):
    grouped = rewards.view(batch_size, num_candidates)
    adv = (grouped - grouped.mean(dim=1, keepdim=True)) / grouped.std(dim=1, keepdim=True).clamp_min(1e-4)
    return adv.clamp(-adv_clip_max, adv_clip_max).reshape(-1)


def main():
    args = parse_args()
    device = get_device()
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

    model, ae, text_encoder, model_info = load_flux2_components(args.model_name, device, debug_mode=args.debug_model)
    model, lora_config = add_flux2_lora(model, args.sft_lora, adapter_name="default")
    model.add_adapter("old", lora_config)
    model.set_adapter("default")
    default_params = [p for p in model.parameters() if p.requires_grad]
    model.set_adapter("old")
    old_params = [p for p in model.parameters() if p.requires_grad]
    model.set_adapter("default")
    with torch.no_grad():
        for src, tgt in zip(default_params, old_params, strict=True):
            tgt.copy_(src.detach())

    optimizer = torch.optim.AdamW(default_params, lr=args.learning_rate, weight_decay=1e-4)
    reward_scorer = FaceEditRewardScorer(device)
    global_step = 0

    for epoch in range(args.num_epochs):
        iterator = tqdm(dataloader, desc=f"Flux2 edit NFT epoch {epoch}")
        for raw_batch in iterator:
            batch = repeat_batch(raw_batch, args.num_candidates, device)

            model.set_adapter("old")
            with torch.no_grad():
                generated, _ = flux2_sample(
                    model,
                    ae,
                    text_encoder,
                    model_info,
                    prompts=batch["instructions"],
                    height=args.resolution,
                    width=args.resolution,
                    num_steps=args.num_inference_steps,
                    guidance=args.guidance,
                    device=device,
                    source_images=batch["source_images"],
                )
            model.set_adapter("default")

            rewards, reward_details = reward_scorer.score_batch(
                generated=generated,
                source=batch["source_images"],
                target=batch["target_images"],
                masks=batch["masks"],
            )
            advantages = group_advantages(
                rewards,
                raw_batch["source_images"].shape[0],
                args.num_candidates,
                args.adv_clip_max,
            )

            with torch.no_grad():
                ctx, ctx_ids = encode_flux2_prompts(text_encoder, batch["instructions"], model_info, device)
                source_latents = encode_flux2_images(ae, batch["source_images"])
                generated_latents = encode_flux2_images(ae, generated)
                ref_tokens, ref_ids = flux2_ref_tokens(source_latents)
                source_tokens, _ = flux2_image_tokens(source_latents)
                latent_mask = F.interpolate(batch["masks"].float(), size=generated_latents.shape[-2:], mode="nearest")
                mask_tokens, _ = flux2_image_tokens(latent_mask.expand_as(generated_latents).to(torch.bfloat16))
                preserve_weight = (1.0 - mask_tokens.float().abs().mean(dim=-1, keepdim=True)).clamp(0, 1)

            t = sample_flux2_timestep(generated_latents.shape[0], device)
            x_tokens, x_ids, _ = make_flux2_noisy_tokens(
                generated_latents,
                t,
                source_latents=source_latents,
                source_mix=args.source_mix,
            )
            generated_tokens, _ = flux2_image_tokens(generated_latents)

            model.set_adapter("old")
            with torch.no_grad():
                old_prediction = flux2_predict(model, x_tokens, x_ids, t, ctx, ctx_ids, args.guidance, ref_tokens, ref_ids).detach()

            model.set_adapter("default")
            forward_prediction = flux2_predict(model, x_tokens, x_ids, t, ctx, ctx_ids, args.guidance, ref_tokens, ref_ids)

            with torch.no_grad():
                with model.disable_adapter():
                    ref_prediction = flux2_predict(model, x_tokens, x_ids, t, ctx, ctx_ids, args.guidance, ref_tokens, ref_ids)

            r = ((advantages / args.adv_clip_max) / 2.0 + 0.5).clamp(0, 1).view(-1, 1)
            positive_prediction = args.nft_beta * forward_prediction + (1.0 - args.nft_beta) * old_prediction
            negative_prediction = (1.0 + args.nft_beta) * old_prediction - args.nft_beta * forward_prediction
            t_tok = t.view(-1, 1, 1).to(torch.bfloat16)

            positive_x0 = x_tokens - t_tok * positive_prediction
            negative_x0 = x_tokens - t_tok * negative_prediction
            positive_loss = (positive_x0.float() - generated_tokens.float()).pow(2).mean(dim=(1, 2))
            negative_loss = (negative_x0.float() - generated_tokens.float()).pow(2).mean(dim=(1, 2))
            policy_loss = (r.flatten() * positive_loss / args.nft_beta + (1.0 - r.flatten()) * negative_loss / args.nft_beta).mean()
            kl_loss = (forward_prediction.float() - ref_prediction.float()).pow(2).mean()
            pred_x0 = x_tokens - t_tok * forward_prediction
            preserve_loss = ((pred_x0.float() - source_tokens.float()).pow(2) * preserve_weight).mean()
            loss = policy_loss + args.kl_beta * kl_loss + args.preserve_beta * preserve_loss

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(default_params, 1.0)
            optimizer.step()

            with torch.no_grad():
                model.set_adapter("old")
                for src, tgt in zip(default_params, old_params, strict=True):
                    tgt.copy_(tgt.detach() * args.old_decay + src.detach() * (1.0 - args.old_decay))
                model.set_adapter("default")

            iterator.set_postfix(
                {
                    "loss": float(loss.detach()),
                    "reward": float(rewards.mean().detach()),
                    **{f"r_{k}": float(v.mean().detach()) for k, v in reward_details.items()},
                }
            )

            global_step += 1
            if args.save_steps > 0 and global_step % args.save_steps == 0:
                model.save_pretrained(flux2_checkpoint_dir(args.output_dir, f"checkpoint-{global_step}"))

    model.save_pretrained(flux2_checkpoint_dir(args.output_dir, "final"))


if __name__ == "__main__":
    main()

