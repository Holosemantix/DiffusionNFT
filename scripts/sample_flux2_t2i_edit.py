# SPDX-License-Identifier: Apache-2.0
"""Run Flux2 klein T2I or edit inference with an optional LoRA checkpoint."""

import argparse
import os

import torch
from PIL import Image
from torchvision.transforms import ToTensor

from flow_grpo.flux2_train_utils import (
    FLUX2_KLEIN_4B,
    add_flux2_lora,
    flux2_sample,
    load_flux2_components,
)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_name", default=FLUX2_KLEIN_4B)
    parser.add_argument("--lora_path", default=None)
    parser.add_argument("--prompt", required=True)
    parser.add_argument("--source_image", default=None)
    parser.add_argument("--output", default="output/flux2_sample.png")
    parser.add_argument("--height", type=int, default=512)
    parser.add_argument("--width", type=int, default=512)
    parser.add_argument("--num_steps", type=int, default=4)
    parser.add_argument("--guidance", type=float, default=1.0)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--debug_model", action="store_true")
    return parser.parse_args()


def main():
    args = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model, ae, text_encoder, model_info = load_flux2_components(args.model_name, device, debug_mode=args.debug_model)
    if args.lora_path:
        model, _ = add_flux2_lora(model, args.lora_path)
    model.eval()

    source = None
    if args.source_image:
        image = Image.open(args.source_image).convert("RGB").resize((args.width, args.height), Image.Resampling.BICUBIC)
        source = ToTensor()(image).unsqueeze(0).to(device)

    with torch.no_grad():
        images, _ = flux2_sample(
            model,
            ae,
            text_encoder,
            model_info,
            prompts=[args.prompt],
            height=args.height,
            width=args.width,
            num_steps=args.num_steps,
            guidance=args.guidance,
            device=device,
            source_images=source,
            seed=args.seed,
        )

    os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)
    arr = (images[0].detach().cpu().permute(1, 2, 0).numpy() * 255).round().clip(0, 255).astype("uint8")
    Image.fromarray(arr).save(args.output)
    print(f"saved {args.output}")


if __name__ == "__main__":
    main()

