import os
from typing import Optional, Sequence, Tuple

import torch
from peft import LoraConfig, PeftModel, get_peft_model


FLUX2_KLEIN_4B = "flux.2-klein-4b"
FLUX2_KLEIN_4B_HF = "black-forest-labs/FLUX.2-klein-4B"
FLUX2_KLEIN_4B_BASE = "flux.2-klein-base-4b"

FLUX2_LORA_TARGET_MODULES = [
    "img_in",
    "txt_in",
    "linear1",
    "linear2",
]


def load_flux2_components(model_name: str, device: torch.device, debug_mode: bool = False):
    try:
        from flux2.util import FLUX2_MODEL_INFO, load_ae, load_flow_model, load_text_encoder
    except ImportError as exc:
        raise ImportError(
            "Flux2 training requires the official BFL flux2 package. Install it with "
            "`pip install git+https://github.com/black-forest-labs/flux2.git` or clone "
            "that repo and run with `PYTHONPATH=/path/to/flux2/src`."
        ) from exc

    model_name = model_name.lower()
    model_info = FLUX2_MODEL_INFO[model_name]
    text_encoder = load_text_encoder(model_name, device=device)
    model = load_flow_model(model_name, debug_mode=debug_mode, device=device)
    ae = load_ae(model_name, device=device)
    text_encoder.eval()
    ae.eval()
    model.train()
    return model, ae, text_encoder, model_info


def add_flux2_lora(model, lora_path: Optional[str] = None, adapter_name: str = "default"):
    config = LoraConfig(
        r=32,
        lora_alpha=64,
        init_lora_weights="gaussian",
        target_modules=FLUX2_LORA_TARGET_MODULES,
    )
    if lora_path:
        model = PeftModel.from_pretrained(model, lora_path, adapter_name=adapter_name, is_trainable=True)
        model.set_adapter(adapter_name)
    else:
        model = get_peft_model(model, config, adapter_name=adapter_name)
    return model, config


@torch.no_grad()
def encode_flux2_prompts(text_encoder, prompts: Sequence[str], model_info: dict, device: torch.device):
    from flux2.sampling import batched_prc_txt

    del model_info
    ctx = text_encoder(list(prompts)).to(torch.bfloat16)
    ctx, ctx_ids = batched_prc_txt(ctx)
    return ctx.to(device), ctx_ids.to(device)


@torch.no_grad()
def encode_flux2_images(ae, images: torch.Tensor) -> torch.Tensor:
    images = images.to(device=next(ae.parameters()).device, dtype=torch.bfloat16) * 2.0 - 1.0
    return ae.encode(images)[0]


@torch.no_grad()
def decode_flux2_latents(ae, latents: torch.Tensor) -> torch.Tensor:
    decoded = ae.decode(latents.to(device=next(ae.parameters()).device, dtype=torch.bfloat16)).float()
    return (decoded / 2.0 + 0.5).clamp(0, 1)


def flux2_image_tokens(latents: torch.Tensor, t_coord: Optional[torch.Tensor] = None):
    from flux2.sampling import batched_prc_img

    tokens, ids = batched_prc_img(latents, t_coord=t_coord)
    return tokens, ids


def flux2_ref_tokens(latents: torch.Tensor, ref_time: int = 10):
    t_coord = torch.full((latents.shape[0], 1), ref_time, device=latents.device, dtype=torch.long)
    return flux2_image_tokens(latents, t_coord=t_coord)


def sample_flux2_timestep(batch_size: int, device: torch.device, min_t: float = 0.02, max_t: float = 0.98):
    return torch.empty(batch_size, device=device, dtype=torch.bfloat16).uniform_(min_t, max_t)


def make_flux2_noisy_tokens(
    target_latents: torch.Tensor,
    t: torch.Tensor,
    source_latents: Optional[torch.Tensor] = None,
    source_mix: float = 0.0,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    noise = torch.randn_like(target_latents, dtype=torch.bfloat16)
    if source_latents is not None and source_mix > 0:
        endpoint = (1.0 - source_mix) * noise + source_mix * source_latents.to(torch.bfloat16)
    else:
        endpoint = noise
    t_expanded = t.view(-1, *([1] * (target_latents.ndim - 1))).to(torch.bfloat16)
    xt = (1.0 - t_expanded) * target_latents.to(torch.bfloat16) + t_expanded * endpoint
    target_v = endpoint - target_latents.to(torch.bfloat16)
    x_tokens, x_ids = flux2_image_tokens(xt)
    v_tokens, _ = flux2_image_tokens(target_v)
    return x_tokens, x_ids, v_tokens


def flux2_predict(
    model,
    x_tokens: torch.Tensor,
    x_ids: torch.Tensor,
    t: torch.Tensor,
    ctx: torch.Tensor,
    ctx_ids: torch.Tensor,
    guidance: float,
    ref_tokens: Optional[torch.Tensor] = None,
    ref_ids: Optional[torch.Tensor] = None,
):
    guidance_vec = torch.full((x_tokens.shape[0],), guidance, device=x_tokens.device, dtype=x_tokens.dtype)
    model_x = x_tokens
    model_ids = x_ids
    if ref_tokens is not None:
        if ref_ids is None:
            raise ValueError("ref_ids is required when ref_tokens is provided")
        model_x = torch.cat([x_tokens, ref_tokens], dim=1)
        model_ids = torch.cat([x_ids, ref_ids], dim=1)
    pred = model(
        x=model_x,
        x_ids=model_ids,
        timesteps=t.to(dtype=x_tokens.dtype),
        ctx=ctx,
        ctx_ids=ctx_ids,
        guidance=guidance_vec,
    )
    return pred[:, : x_tokens.shape[1]]


def flux2_checkpoint_dir(output_dir: str, step_name: str) -> str:
    path = os.path.join(output_dir, step_name, "lora")
    os.makedirs(path, exist_ok=True)
    return path


@torch.no_grad()
def flux2_sample(
    model,
    ae,
    text_encoder,
    model_info: dict,
    prompts: Sequence[str],
    height: int,
    width: int,
    num_steps: int,
    guidance: float,
    device: torch.device,
    source_images: Optional[torch.Tensor] = None,
    seed: Optional[int] = None,
):
    from flux2.sampling import get_schedule, scatter_ids

    ctx, ctx_ids = encode_flux2_prompts(text_encoder, prompts, model_info, device)
    batch_size = len(prompts)
    generator = torch.Generator(device=device)
    if seed is not None:
        generator.manual_seed(seed)
    noise = torch.randn(
        (batch_size, 128, height // 16, width // 16),
        generator=generator,
        dtype=torch.bfloat16,
        device=device,
    )
    x_tokens, x_ids = flux2_image_tokens(noise)

    ref_tokens = None
    ref_ids = None
    if source_images is not None:
        source_latents = encode_flux2_images(ae, source_images)
        ref_tokens, ref_ids = flux2_ref_tokens(source_latents)

    timesteps = get_schedule(num_steps, x_tokens.shape[1])
    for t_curr, t_prev in zip(timesteps[:-1], timesteps[1:]):
        t = torch.full((batch_size,), t_curr, dtype=torch.bfloat16, device=device)
        pred = flux2_predict(model, x_tokens, x_ids, t, ctx, ctx_ids, guidance, ref_tokens, ref_ids)
        x_tokens = x_tokens + (t_prev - t_curr) * pred

    latents = torch.cat(scatter_ids(x_tokens, x_ids)).squeeze(2)
    images = decode_flux2_latents(ae, latents)
    return images, latents
