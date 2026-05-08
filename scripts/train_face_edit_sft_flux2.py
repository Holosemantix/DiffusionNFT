# SPDX-License-Identifier: Apache-2.0
"""Flux2 klein 4B edit SFT for small low-resolution face restoration.

The important Flux2 edit contract is:
  current sequence = noisy target image tokens
  conditioning sequence = source/edit image tokens
  transformer input = token concat(current, conditioning)

Only the current noisy target tokens are supervised.
"""

import argparse
import os

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm

from flow_grpo.device_utils import get_device, get_pin_memory
from flow_grpo.editing_data import FaceEditDataset, collate_edit_samples
from flow_grpo.flux2_train_utils import (
    FLUX2_KLEIN_4B,
    add_flux2_lora,
    configure_flux2_local_paths,
    enable_flux2_gradient_checkpointing,
    encode_flux2_images,
    encode_flux2_prompts,
    flux2_checkpoint_dir,
    flux2_image_tokens,
    flux2_predict,
    flux2_ref_tokens,
    load_flux2_components,
    make_flux2_noisy_tokens,
    sample_flux2_timestep,
)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_name", default=FLUX2_KLEIN_4B)
    parser.add_argument("--flux2_local_dir", default=None, help="Directory containing local FLUX.2 safetensors files")
    parser.add_argument("--flux2_model_path", default=None, help="Local FLUX.2 transformer safetensors path")
    parser.add_argument("--flux2_ae_path", default=None, help="Local FLUX.2 autoencoder safetensors path")
    parser.add_argument("--flux2_text_encoder_path", default=None, help="Local FLUX.2 text encoder directory")
    parser.add_argument("--flux2_tokenizer_path", default=None, help="Local FLUX.2 tokenizer directory")
    parser.add_argument("--dataset_source", default="wider_face_restore", choices=["wider_face_restore", "magicbrush", "canonical_jsonl"])
    parser.add_argument("--dataset_split", default="train")
    parser.add_argument("--jsonl_path", default=None)
    parser.add_argument("--cache_dir", default=None)
    parser.add_argument("--output_dir", default="logs/face_edit/sft_flux2")
    parser.add_argument("--resume_lora", default=None)
    parser.add_argument("--resolution", type=int, default=512)
    parser.add_argument("--max_samples", type=int, default=None)
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--num_epochs", type=int, default=1)
    parser.add_argument("--learning_rate", type=float, default=1e-4)
    parser.add_argument("--gradient_accumulation_steps", type=int, default=4)
    parser.add_argument("--guidance", type=float, default=1.0)
    parser.add_argument("--source_mix", type=float, default=0.25)
    parser.add_argument("--save_steps", type=int, default=500)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--debug_model", action="store_true")
    parser.add_argument(
        "--gradient_checkpointing",
        action="store_true",
        help="Wrap flux2 transformer block forwards with torch.utils.checkpoint to "
             "trade compute for activation memory. Strongly recommended at 1024 res.",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    configure_flux2_local_paths(
        args.model_name,
        local_dir=args.flux2_local_dir,
        model_path=args.flux2_model_path,
        ae_path=args.flux2_ae_path,
        text_encoder_path=args.flux2_text_encoder_path,
        tokenizer_path=args.flux2_tokenizer_path,
    )
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
    model, _ = add_flux2_lora(model, args.resume_lora)
    if args.gradient_checkpointing:
        n_patched = enable_flux2_gradient_checkpointing(model)
        print(f"[flux2] gradient checkpointing enabled on {n_patched} transformer blocks")
    trainable_params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(trainable_params, lr=args.learning_rate, weight_decay=1e-4)

    global_step = 0
    optimizer.zero_grad()
    for epoch in range(args.num_epochs):
        iterator = tqdm(dataloader, desc=f"Flux2 edit SFT epoch {epoch}")
        for batch in iterator:
            source = batch["source_images"].to(device)
            target = batch["target_images"].to(device)
            masks = batch["masks"].to(device)
            prompts = batch["instructions"]

            with torch.no_grad():
                ctx, ctx_ids = encode_flux2_prompts(text_encoder, prompts, model_info, device)
                source_latents = encode_flux2_images(ae, source)
                target_latents = encode_flux2_images(ae, target)
                ref_tokens, ref_ids = flux2_ref_tokens(source_latents)

            t = sample_flux2_timestep(target_latents.shape[0], device)
            x_tokens, x_ids, v_tokens = make_flux2_noisy_tokens(
                target_latents,
                t,
                source_latents=source_latents,
                source_mix=args.source_mix,
            )
            pred = flux2_predict(model, x_tokens, x_ids, t, ctx, ctx_ids, args.guidance, ref_tokens, ref_ids)

            latent_mask = F.interpolate(masks.float(), size=target_latents.shape[-2:], mode="nearest")
            mask_tokens, _ = flux2_image_tokens(latent_mask.expand_as(target_latents).to(torch.bfloat16))
            token_weight = mask_tokens.float().abs().mean(dim=-1, keepdim=True).clamp(0.2, 1.0)
            edit_loss = ((pred.float() - v_tokens.float()).pow(2) * token_weight).mean()
            loss = edit_loss / args.gradient_accumulation_steps
            loss.backward()

            if (global_step + 1) % args.gradient_accumulation_steps == 0:
                torch.nn.utils.clip_grad_norm_(trainable_params, 1.0)
                optimizer.step()
                optimizer.zero_grad()

            iterator.set_postfix({"loss": float(edit_loss.detach())})
            global_step += 1
            if args.save_steps > 0 and global_step % args.save_steps == 0:
                model.save_pretrained(flux2_checkpoint_dir(args.output_dir, f"checkpoint-{global_step}"))

    model.save_pretrained(flux2_checkpoint_dir(args.output_dir, "final"))


if __name__ == "__main__":
    main()
