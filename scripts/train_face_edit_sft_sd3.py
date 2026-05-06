# SPDX-License-Identifier: Apache-2.0
"""SFT warm-start for small-face image editing with SD3.5 LoRA.

The model is trained on canonical edit samples:
source image + instruction + edit mask -> target image.

For small-face restoration, `wider_face_restore` creates paired data by
degrading annotated small face regions and using the original image as target.
"""

import argparse
import os
from collections import defaultdict

import torch
import torch.nn.functional as F
from diffusers import StableDiffusion3Pipeline
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
from tqdm import tqdm

from flow_grpo.device_utils import get_device, get_grad_scaler, get_amp_context, get_pin_memory
from flow_grpo.editing_data import FaceEditDataset, collate_edit_samples
from flow_grpo.face_edit_losses import robust_edit_supervision_loss
from flow_grpo.sd3_edit_train_utils import (
    SD3_MODEL_ID,
    add_sd3_lora,
    compute_prompt_embeddings,
    decode_vae_latents,
    encode_vae_images,
    is_main_process,
    make_edit_flow_noisy_latents,
    sample_flow_timestep,
    setup_distributed,
    unwrap_model,
)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset_source", default="wider_face_restore", choices=["wider_face_restore", "magicbrush", "canonical_jsonl"])
    parser.add_argument("--dataset_split", default="train")
    parser.add_argument("--jsonl_path", default=None)
    parser.add_argument("--cache_dir", default=None)
    parser.add_argument("--output_dir", default="logs/face_edit/sft_sd3")
    parser.add_argument("--pretrained_model", default=SD3_MODEL_ID)
    parser.add_argument("--resume_lora", default=None)
    parser.add_argument("--resolution", type=int, default=512)
    parser.add_argument("--max_samples", type=int, default=None)
    parser.add_argument("--train_batch_size", type=int, default=1)
    parser.add_argument("--num_epochs", type=int, default=1)
    parser.add_argument("--learning_rate", type=float, default=1e-4)
    parser.add_argument("--gradient_accumulation_steps", type=int, default=4)
    parser.add_argument("--max_grad_norm", type=float, default=1.0)
    parser.add_argument("--mixed_precision", default="fp16", choices=["fp16", "bf16", "no"])
    parser.add_argument("--source_mix", type=float, default=0.35)
    parser.add_argument("--edit_loss_weight", type=float, default=1.0)
    parser.add_argument("--preserve_loss_weight", type=float, default=0.25)
    parser.add_argument("--image_loss_weight", type=float, default=0.05)
    parser.add_argument("--save_steps", type=int, default=500)
    parser.add_argument("--num_workers", type=int, default=4)
    return parser.parse_args()


def main():
    args = parse_args()
    rank, world_size, local_rank = setup_distributed()
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
    sampler = DistributedSampler(dataset, num_replicas=world_size, rank=rank, shuffle=True) if world_size > 1 else None
    dataloader = DataLoader(
        dataset,
        batch_size=args.train_batch_size,
        sampler=sampler,
        shuffle=sampler is None,
        num_workers=args.num_workers,
        collate_fn=collate_edit_samples,
        pin_memory=get_pin_memory(device),
    )

    pipeline = StableDiffusion3Pipeline.from_pretrained(args.pretrained_model)
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

    transformer, _ = add_sd3_lora(pipeline.transformer.to(device), args.resume_lora)
    transformer.train()
    trainable_params = [p for p in transformer.parameters() if p.requires_grad]
    if world_size > 1:
        transformer = DDP(transformer, device_ids=[local_rank], output_device=local_rank, find_unused_parameters=False)

    optimizer = torch.optim.AdamW(trainable_params, lr=args.learning_rate, weight_decay=1e-4)
    scaler = get_grad_scaler(device.type, enabled=args.mixed_precision == "fp16")
    text_encoders = [pipeline.text_encoder, pipeline.text_encoder_2, pipeline.text_encoder_3]
    tokenizers = [pipeline.tokenizer, pipeline.tokenizer_2, pipeline.tokenizer_3]

    global_step = 0
    optimizer.zero_grad()

    for epoch in range(args.num_epochs):
        if sampler is not None:
            sampler.set_epoch(epoch)
        iterator = tqdm(dataloader, disable=not is_main_process(rank), desc=f"SFT epoch {epoch}")
        for batch in iterator:
            source = batch["source_images"].to(device)
            target = batch["target_images"].to(device)
            masks = batch["masks"].to(device)
            prompts = batch["instructions"]

            with torch.no_grad():
                prompt_embeds, pooled_embeds = compute_prompt_embeddings(prompts, text_encoders, tokenizers, 128, device)
                source_latents = encode_vae_images(pipeline.vae, source, torch.float32)
                target_latents = encode_vae_images(pipeline.vae, target, torch.float32)

            t = sample_flow_timestep(target_latents.shape[0], device)
            xt, target_v = make_edit_flow_noisy_latents(target_latents, source_latents, t, source_mix=args.source_mix)
            timesteps = (t * 1000).long()

            with get_amp_context(device.type, enabled=amp_enabled, dtype=dtype):
                pred_v = transformer(
                    hidden_states=xt.to(dtype),
                    timestep=timesteps,
                    encoder_hidden_states=prompt_embeds,
                    pooled_projections=pooled_embeds,
                    return_dict=False,
                )[0].float()

                latent_mask = F.interpolate(masks.float(), size=target_latents.shape[-2:], mode="nearest")
                latent_mask = latent_mask.expand_as(target_latents)
                edit_loss = F.mse_loss(pred_v * latent_mask, target_v * latent_mask)

                t_expanded = t.view(-1, *([1] * (target_latents.ndim - 1)))
                pred_x0 = xt.float() - t_expanded * pred_v
                preserve_loss = F.smooth_l1_loss(pred_x0 * (1 - latent_mask), source_latents.float() * (1 - latent_mask))
                loss = args.edit_loss_weight * edit_loss + args.preserve_loss_weight * preserve_loss

                image_loss = torch.zeros((), device=device)
                image_terms = {}
                if args.image_loss_weight > 0:
                    pred_image = decode_vae_latents(pipeline.vae, pred_x0)
                    image_loss, image_terms = robust_edit_supervision_loss(
                        pred_image,
                        target,
                        source,
                        masks,
                        edit_weight=1.0,
                        preserve_weight=0.5,
                        global_weight=0.1,
                    )
                    loss = loss + args.image_loss_weight * image_loss

            loss_to_backprop = loss / args.gradient_accumulation_steps
            if args.mixed_precision == "fp16":
                scaler.scale(loss_to_backprop).backward()
            else:
                loss_to_backprop.backward()

            if (global_step + 1) % args.gradient_accumulation_steps == 0:
                if args.mixed_precision == "fp16":
                    scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(trainable_params, args.max_grad_norm)
                if args.mixed_precision == "fp16":
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    optimizer.step()
                optimizer.zero_grad()

            if is_main_process(rank):
                logs = {
                    "loss": float(loss.detach()),
                    "edit_loss": float(edit_loss.detach()),
                    "preserve_loss": float(preserve_loss.detach()),
                    "image_loss": float(image_loss.detach()),
                    **{k: float(v) for k, v in image_terms.items()},
                }
                iterator.set_postfix(logs)

            global_step += 1
            if is_main_process(rank) and args.save_steps > 0 and global_step % args.save_steps == 0:
                save_dir = os.path.join(args.output_dir, f"checkpoint-{global_step}", "lora")
                os.makedirs(save_dir, exist_ok=True)
                unwrap_model(transformer).save_pretrained(save_dir)

    if is_main_process(rank):
        save_dir = os.path.join(args.output_dir, "final", "lora")
        os.makedirs(save_dir, exist_ok=True)
        unwrap_model(transformer).save_pretrained(save_dir)

    if world_size > 1:
        torch.distributed.destroy_process_group()


if __name__ == "__main__":
    main()

