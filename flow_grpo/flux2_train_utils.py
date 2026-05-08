import hashlib
import os
import re
import tempfile
from typing import Optional, Sequence, Tuple

import torch


FLUX2_KLEIN_4B = "flux.2-klein-4b"
FLUX2_KLEIN_4B_HF = "black-forest-labs/FLUX.2-klein-4B"
FLUX2_KLEIN_4B_BASE = "flux.2-klein-base-4b"

FLUX2_LOCAL_MODEL_FILES = {
    FLUX2_KLEIN_4B: ("KLEIN_4B_MODEL_PATH", "flux-2-klein-4b.safetensors"),
    FLUX2_KLEIN_4B_BASE: ("KLEIN_4B_BASE_MODEL_PATH", "flux-2-klein-base-4b.safetensors"),
}

FLUX2_LORA_TARGET_MODULES = [
    "img_in",
    "txt_in",
    "linear1",
    "linear2",
]


def patch_torch_pytree_for_transformers() -> None:
    """Bridge torch 2.1 private pytree API name expected by newer transformers."""

    try:
        import torch.utils._pytree as pytree
    except Exception:
        return

    if not hasattr(pytree, "register_pytree_node") and hasattr(pytree, "_register_pytree_node"):

        def register_pytree_node(type_, flatten_fn, unflatten_fn, *args, **kwargs):
            kwargs.pop("serialized_type_name", None)
            kwargs.pop("to_dumpable_context", None)
            kwargs.pop("from_dumpable_context", None)
            return pytree._register_pytree_node(type_, flatten_fn, unflatten_fn, *args, **kwargs)

        pytree.register_pytree_node = register_pytree_node


def configure_flux2_local_paths(
    model_name: str,
    local_dir: Optional[str] = None,
    model_path: Optional[str] = None,
    ae_path: Optional[str] = None,
    text_encoder_path: Optional[str] = None,
    tokenizer_path: Optional[str] = None,
) -> None:
    """Set official flux2 loader env vars for locally downloaded weights."""

    model_name = model_name.lower()
    if text_encoder_path is None and local_dir:
        candidate = os.path.join(local_dir, "text_encoder")
        if os.path.isdir(candidate):
            text_encoder_path = candidate
    if tokenizer_path is None and local_dir:
        candidate = os.path.join(local_dir, "tokenizer")
        if os.path.isdir(candidate):
            tokenizer_path = candidate
    if text_encoder_path:
        if not os.path.exists(text_encoder_path):
            raise FileNotFoundError(f"Flux2 text encoder path does not exist: {text_encoder_path}")
        os.environ["FLUX2_TEXT_ENCODER_PATH"] = os.path.abspath(text_encoder_path)
    if tokenizer_path:
        if not os.path.exists(tokenizer_path):
            raise FileNotFoundError(f"Flux2 tokenizer path does not exist: {tokenizer_path}")
        os.environ["FLUX2_TOKENIZER_PATH"] = os.path.abspath(tokenizer_path)

    if model_name not in FLUX2_LOCAL_MODEL_FILES:
        if model_path:
            os.environ["FLUX2_MODEL_PATH"] = os.path.abspath(model_path)
        elif local_dir:
            candidate = os.path.join(local_dir, "flux2-dev.safetensors")
            if os.path.exists(candidate):
                os.environ["FLUX2_MODEL_PATH"] = os.path.abspath(candidate)
        if ae_path:
            os.environ["AE_MODEL_PATH"] = _resolve_ae_env_path(ae_path)
        elif local_dir:
            candidate = os.path.join(local_dir, "ae.safetensors")
            if os.path.exists(candidate):
                os.environ["AE_MODEL_PATH"] = _resolve_ae_env_path(candidate)
        return

    model_env, default_model_file = FLUX2_LOCAL_MODEL_FILES[model_name]
    if model_path is None and local_dir:
        model_path = os.path.join(local_dir, default_model_file)
    if ae_path is None and local_dir:
        ae_path = os.path.join(local_dir, "ae.safetensors")

    if model_path:
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Flux2 model file does not exist: {model_path}")
        os.environ[model_env] = os.path.abspath(model_path)
    if ae_path:
        if not os.path.exists(ae_path):
            raise FileNotFoundError(f"Flux2 autoencoder file does not exist: {ae_path}")
        os.environ["AE_MODEL_PATH"] = _resolve_ae_env_path(ae_path)


_DIFFUSERS_AE_MARKERS = ("down_blocks", "up_blocks", "mid_block", "conv_norm_out")


def _convert_diffusers_ae_state_dict(sd: dict) -> dict:
    """Remap a diffusers-format AutoencoderKL state_dict to the native ldm/flux2 layout."""

    out: dict = {}
    # Decoder up_blocks index is reversed relative to native up.* (mid → up_blocks.0 ↔ up.3, last ↔ up.0).
    NUM_UP = 4

    def _attn_reshape(key: str, val: torch.Tensor) -> torch.Tensor:
        # diffusers stores mid attention q/k/v/to_out.0 as nn.Linear (2D); ldm uses 1x1 Conv2d (4D).
        if key.endswith(".weight") and val.ndim == 2:
            return val.unsqueeze(-1).unsqueeze(-1).contiguous()
        return val

    for k, v in sd.items():
        nk = k

        # Top-level quant convs live under encoder/decoder in native layout.
        if k.startswith("quant_conv."):
            nk = "encoder.quant_conv." + k[len("quant_conv."):]
            out[nk] = v
            continue
        if k.startswith("post_quant_conv."):
            nk = "decoder.post_quant_conv." + k[len("post_quant_conv."):]
            out[nk] = v
            continue

        # conv_norm_out → norm_out (encoder + decoder).
        nk = nk.replace(".conv_norm_out.", ".norm_out.")

        # Mid block resnets/attention.
        nk = nk.replace(".mid_block.resnets.0.", ".mid.block_1.")
        nk = nk.replace(".mid_block.resnets.1.", ".mid.block_2.")
        if ".mid_block.attentions.0." in nk:
            nk = nk.replace(".mid_block.attentions.0.group_norm.", ".mid.attn_1.norm.")
            nk = nk.replace(".mid_block.attentions.0.to_q.", ".mid.attn_1.q.")
            nk = nk.replace(".mid_block.attentions.0.to_k.", ".mid.attn_1.k.")
            nk = nk.replace(".mid_block.attentions.0.to_v.", ".mid.attn_1.v.")
            nk = nk.replace(".mid_block.attentions.0.to_out.0.", ".mid.attn_1.proj_out.")
            v = _attn_reshape(nk, v)

        # Encoder down_blocks.{i}.resnets.{j} → encoder.down.{i}.block.{j}
        m = re.match(r"^encoder\.down_blocks\.(\d+)\.resnets\.(\d+)\.(.+)$", nk)
        if m:
            i, j, rest = m.group(1), m.group(2), m.group(3)
            rest = rest.replace("conv_shortcut", "nin_shortcut")
            nk = f"encoder.down.{i}.block.{j}.{rest}"
        else:
            m = re.match(r"^encoder\.down_blocks\.(\d+)\.downsamplers\.0\.(.+)$", nk)
            if m:
                nk = f"encoder.down.{m.group(1)}.downsample.{m.group(2)}"

        # Decoder up_blocks.{i} → up.{NUM_UP-1-i}
        m = re.match(r"^decoder\.up_blocks\.(\d+)\.resnets\.(\d+)\.(.+)$", nk)
        if m:
            i = int(m.group(1))
            j, rest = m.group(2), m.group(3)
            rest = rest.replace("conv_shortcut", "nin_shortcut")
            nk = f"decoder.up.{NUM_UP - 1 - i}.block.{j}.{rest}"
        else:
            m = re.match(r"^decoder\.up_blocks\.(\d+)\.upsamplers\.0\.(.+)$", nk)
            if m:
                i = int(m.group(1))
                nk = f"decoder.up.{NUM_UP - 1 - i}.upsample.{m.group(2)}"

        out[nk] = v

    return out


def _looks_like_diffusers_ae(sd: dict) -> bool:
    return any(any(marker in k for marker in _DIFFUSERS_AE_MARKERS) for k in sd.keys())


def _maybe_convert_ae_checkpoint(ae_path: str) -> str:
    """If `ae_path` is a diffusers-format AE checkpoint, convert and cache to temp file.

    Returns the path that should be passed to flux2's load_ae (native ldm layout).
    """

    from safetensors import safe_open
    from safetensors.torch import save_file

    if not ae_path.endswith(".safetensors"):
        return ae_path

    # Peek at keys without loading full tensors.
    with safe_open(ae_path, framework="pt") as f:
        keys = list(f.keys())
    if not any(any(marker in k for marker in _DIFFUSERS_AE_MARKERS) for k in keys):
        return ae_path

    # Cache converted file alongside source by content-hash so repeated runs are cheap.
    try:
        st = os.stat(ae_path)
        sig = f"{ae_path}:{st.st_size}:{int(st.st_mtime)}"
    except OSError:
        sig = ae_path
    digest = hashlib.sha1(sig.encode("utf-8")).hexdigest()[:12]
    cache_dir = os.path.join(tempfile.gettempdir(), "flux2_ae_native")
    os.makedirs(cache_dir, exist_ok=True)
    cached = os.path.join(cache_dir, f"ae_native_{digest}.safetensors")
    if os.path.exists(cached):
        return cached

    sd = {}
    with safe_open(ae_path, framework="pt") as f:
        for k in f.keys():
            sd[k] = f.get_tensor(k)
    converted = _convert_diffusers_ae_state_dict(sd)
    tmp = cached + ".tmp"
    save_file(converted, tmp)
    os.replace(tmp, cached)
    print(f"[flux2] converted diffusers AE checkpoint to native layout: {cached}")
    return cached


def _resolve_ae_env_path(ae_path: str) -> str:
    abs_path = os.path.abspath(ae_path)
    try:
        return _maybe_convert_ae_checkpoint(abs_path)
    except Exception as exc:  # pragma: no cover - best-effort fallback
        print(f"[flux2] AE auto-conversion skipped ({type(exc).__name__}: {exc}); using {abs_path}")
        return abs_path


def _merge_text_encoder_and_tokenizer_dirs(text_encoder_path: str, tokenizer_path: Optional[str]) -> str:
    if not tokenizer_path:
        return text_encoder_path

    merged_dir = tempfile.mkdtemp(prefix="flux2_text_encoder_")
    for source_dir in (text_encoder_path, tokenizer_path):
        for name in os.listdir(source_dir):
            source = os.path.join(source_dir, name)
            target = os.path.join(merged_dir, name)
            if not os.path.exists(target):
                os.symlink(source, target)
    return merged_dir


def _load_flux2_text_encoder(model_name: str, device: torch.device, load_text_encoder):
    text_encoder_path = os.environ.get("FLUX2_TEXT_ENCODER_PATH")
    if not text_encoder_path:
        return load_text_encoder(model_name, device=device)

    from flux2.text_encoder import Qwen3Embedder

    model_spec = _merge_text_encoder_and_tokenizer_dirs(text_encoder_path, os.environ.get("FLUX2_TOKENIZER_PATH"))

    # transformers 4.56+ meta-tensor load path triggers CUDA init when target
    # device is non-CPU/non-CUDA (e.g. npu). Build the embedder on CPU first and
    # move the underlying model to the requested device afterwards.
    if device.type not in ("cpu", "cuda"):
        embedder = Qwen3Embedder(model_spec=model_spec, device=torch.device("cpu"))
        if hasattr(embedder, "model"):
            embedder.model = embedder.model.to(device)
        if hasattr(embedder, "device"):
            try:
                embedder.device = device
            except AttributeError:
                pass
        return embedder

    return Qwen3Embedder(model_spec=model_spec, device=device)


def load_flux2_components(model_name: str, device: torch.device, debug_mode: bool = False):
    patch_torch_pytree_for_transformers()
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
    text_encoder = _load_flux2_text_encoder(model_name, device, load_text_encoder)
    model = load_flow_model(model_name, debug_mode=debug_mode, device=device)
    ae = load_ae(model_name, device=device)
    text_encoder.eval()
    ae.eval()
    model.train()
    return model, ae, text_encoder, model_info


def enable_flux2_gradient_checkpointing(model) -> int:
    """Wrap each transformer block's forward methods with torch.utils.checkpoint.

    flux2's training path calls block.forward_kv_extract(...) directly (not
    block(...)), so the standard HF gradient_checkpointing_enable hook misses
    it. We monkey-patch both `forward` and `forward_kv_extract` (when present)
    on each block found under common attribute names.

    Returns the number of blocks patched (for logging).
    """

    from torch.utils.checkpoint import checkpoint as _checkpoint

    base = model.get_base_model() if hasattr(model, "get_base_model") else model

    # PEFT requires inputs to require grad for checkpointing to backprop through LoRA.
    if hasattr(base, "enable_input_require_grads"):
        try:
            base.enable_input_require_grads()
        except Exception:
            pass
    elif hasattr(model, "enable_input_require_grads"):
        try:
            model.enable_input_require_grads()
        except Exception:
            pass

    block_lists = []
    for attr in ("blocks", "transformer_blocks", "single_blocks", "double_blocks", "layers"):
        child = getattr(base, attr, None)
        if isinstance(child, torch.nn.ModuleList) and len(child) > 0:
            block_lists.append(child)

    def _make_wrapper(orig_method):
        def wrapped(*args, **kwargs):
            if not torch.is_grad_enabled():
                return orig_method(*args, **kwargs)

            def closure(*pos):
                return orig_method(*pos, **kwargs)

            return _checkpoint(closure, *args, use_reentrant=False)

        return wrapped

    patched = 0
    for blocks in block_lists:
        for block in blocks:
            for method_name in ("forward_kv_extract", "forward"):
                if hasattr(block, method_name):
                    orig = getattr(block, method_name)
                    setattr(block, method_name, _make_wrapper(orig))
            patched += 1

    if patched == 0:
        raise RuntimeError(
            "enable_flux2_gradient_checkpointing: could not find a transformer block ModuleList "
            f"on {type(base).__name__}. Inspect the model and add the attr name to block_lists."
        )
    return patched


def add_flux2_lora(model, lora_path: Optional[str] = None, adapter_name: str = "default"):
    patch_torch_pytree_for_transformers()
    from peft import LoraConfig, PeftModel, get_peft_model

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
    latents = ae.encode(images)
    # flux2's native AutoEncoder.encode returns the latent tensor directly (B, C, H, W);
    # diffusers' AutoencoderKL returns an output object whose [0] is latent_dist. Support
    # both: only unwrap when the returned object isn't already a tensor.
    if not isinstance(latents, torch.Tensor):
        latents = latents[0]
        if hasattr(latents, "sample"):
            latents = latents.sample()
    return latents


@torch.no_grad()
def decode_flux2_latents(ae, latents: torch.Tensor) -> torch.Tensor:
    decoded = ae.decode(latents.to(device=next(ae.parameters()).device, dtype=torch.bfloat16)).float()
    return (decoded / 2.0 + 0.5).clamp(0, 1)


def flux2_image_tokens(latents: torch.Tensor, t_coord: Optional[torch.Tensor] = None):
    from flux2.sampling import batched_prc_img

    # flux2.sampling.prc_img builds coord tensors with mixed devices (input tensor's
    # device for some, CPU default for others). On CUDA torch.cartesian_prod silently
    # promotes; on NPU it errors with "meshgrid expects all tensors to have the same
    # device". Run prc on CPU and move the (small) outputs back to the input device.
    src_device = latents.device
    if src_device.type not in ("cpu", "cuda"):
        latents_cpu = latents.detach().to("cpu")
        t_coord_cpu = t_coord.detach().to("cpu") if t_coord is not None else None
        tokens, ids = batched_prc_img(latents_cpu, t_coord=t_coord_cpu)
        return tokens.to(src_device), ids.to(src_device)
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
    if device.type == "npu":
        generator = torch.Generator(device="cpu")
        if seed is not None:
            generator.manual_seed(seed)
        noise = torch.randn(
            (batch_size, 128, height // 16, width // 16),
            generator=generator,
            dtype=torch.bfloat16,
            device="cpu",
        ).to(device)
    else:
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
