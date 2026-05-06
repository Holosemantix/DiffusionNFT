import os
from typing import Iterable, List, Tuple

import torch
import torch.distributed as dist
from peft import LoraConfig, PeftModel, get_peft_model

from flow_grpo.diffusers_patch.train_dreambooth_lora_sd3 import encode_prompt


SD3_MODEL_ID = "stabilityai/stable-diffusion-3.5-medium"
SD3_LORA_TARGET_MODULES = [
    "attn.add_k_proj",
    "attn.add_q_proj",
    "attn.add_v_proj",
    "attn.to_add_out",
    "attn.to_k",
    "attn.to_out.0",
    "attn.to_q",
    "attn.to_v",
]


def setup_distributed() -> Tuple[int, int, int]:
    rank = int(os.environ.get("RANK", "0"))
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    if world_size > 1 and not dist.is_initialized():
        os.environ["MASTER_ADDR"] = os.getenv("MASTER_ADDR", "localhost")
        os.environ["MASTER_PORT"] = os.getenv("MASTER_PORT", "12355")
        backend = "hccl" if hasattr(torch, "npu") and torch.npu.is_available() else "nccl"
        dist.init_process_group(backend, rank=rank, world_size=world_size)
    if hasattr(torch, "npu") and torch.npu.is_available():
        torch.npu.set_device(local_rank)
    elif torch.cuda.is_available():
        torch.cuda.set_device(local_rank)
    return rank, world_size, local_rank


def is_main_process(rank: int) -> bool:
    return rank == 0


def unwrap_model(model):
    return model.module if hasattr(model, "module") else model


def add_sd3_lora(transformer, lora_path=None, adapter_name="default"):
    config = LoraConfig(r=32, lora_alpha=64, init_lora_weights="gaussian", target_modules=SD3_LORA_TARGET_MODULES)
    if lora_path:
        transformer = PeftModel.from_pretrained(transformer, lora_path, adapter_name=adapter_name, is_trainable=True)
        transformer.set_adapter(adapter_name)
    else:
        transformer = get_peft_model(transformer, config, adapter_name=adapter_name)
    return transformer, config


@torch.no_grad()
def compute_prompt_embeddings(prompts, text_encoders, tokenizers, max_sequence_length, device):
    prompt_embeds, pooled_prompt_embeds = encode_prompt(text_encoders, tokenizers, prompts, max_sequence_length)
    return prompt_embeds.to(device), pooled_prompt_embeds.to(device)


def encode_vae_images(vae, images: torch.Tensor, dtype: torch.dtype) -> torch.Tensor:
    images = images.to(device=vae.device, dtype=dtype) * 2.0 - 1.0
    latents = vae.encode(images).latent_dist.sample()
    shift = getattr(vae.config, "shift_factor", 0.0)
    scale = getattr(vae.config, "scaling_factor", 1.0)
    return (latents - shift) * scale


def decode_vae_latents(vae, latents: torch.Tensor) -> torch.Tensor:
    scale = getattr(vae.config, "scaling_factor", 1.0)
    shift = getattr(vae.config, "shift_factor", 0.0)
    latents = (latents / scale) + shift
    images = vae.decode(latents.to(dtype=vae.dtype), return_dict=False)[0]
    return (images / 2.0 + 0.5).clamp(0, 1)


def sample_flow_timestep(batch_size: int, device: torch.device, min_t: float = 0.02, max_t: float = 0.98):
    return torch.empty(batch_size, device=device).uniform_(min_t, max_t)


def make_edit_flow_noisy_latents(
    target_latents: torch.Tensor,
    source_latents: torch.Tensor,
    t: torch.Tensor,
    source_mix: float = 0.35,
) -> Tuple[torch.Tensor, torch.Tensor]:
    endpoint = (1.0 - source_mix) * torch.randn_like(target_latents.float()) + source_mix * source_latents.float()
    t_expanded = t.view(-1, *([1] * (target_latents.ndim - 1)))
    xt = (1.0 - t_expanded) * target_latents.float() + t_expanded * endpoint
    target_v = endpoint - target_latents.float()
    return xt, target_v

