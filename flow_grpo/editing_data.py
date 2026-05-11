import io
import json
import math
import os
import random
import zipfile
from dataclasses import dataclass
from glob import glob
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch
from PIL import Image, ImageEnhance, ImageFilter, ImageOps
from torch.utils.data import Dataset

from flow_grpo.face_edit_losses import detect_faces_opencv
from flow_grpo.flux2_train_utils import patch_torch_pytree_for_transformers

patch_torch_pytree_for_transformers()

from torchvision import transforms  # noqa: E402


CANONICAL_FIELDS = {
    "source_image": "PIL RGB image before editing",
    "target_image": "PIL RGB image after editing or restoration",
    "mask_image": "PIL L image, white means editable/evaluated region",
    "instruction": "text instruction for image editing",
    "metadata": "dataset-specific metadata",
}

IMAGE_EXTENSIONS = (".jpg", ".jpeg", ".png", ".webp", ".bmp")

FACE_PRESERVE_PROMPTS = {
    "restore": [
        "[preserve face id] restore small or distant low-resolution faces while preserving the rest of the image",
        "[preserve small faces] recover facial details and keep face identity unchanged",
        "[face_identity_lock] enhance degraded small faces without changing the background",
        "restore the small low-resolution face and preserve facial identity",
    ],
    "background": [
        "[preserve face id] change the background appearance while keeping all faces unchanged",
        "[preserve small faces] edit only non-face regions and keep facial identity fixed",
        "[face_identity_lock] apply a background edit without altering small or distant faces",
        "make a visual edit outside the face region and preserve all face identities",
    ],
    "noop": [
        "[preserve face id] keep the image unchanged and preserve all small faces",
        "[preserve small faces] preserve facial identity and do not edit the image",
        "[face_identity_lock] keep distant faces identical and leave the image unchanged",
    ],
}


@dataclass
class EditSample:
    source_image: Image.Image
    target_image: Image.Image
    mask_image: Image.Image
    instruction: str
    metadata: Dict[str, Any]


def _pil_rgb(image: Any) -> Image.Image:
    if isinstance(image, Image.Image):
        return image.convert("RGB")
    if isinstance(image, bytes):
        return Image.open(io.BytesIO(image)).convert("RGB")
    if isinstance(image, np.ndarray):
        return Image.fromarray(image).convert("RGB")
    if isinstance(image, str):
        return Image.open(image).convert("RGB")
    raise TypeError(f"Unsupported image value: {type(image)!r}")


def _pil_mask(image: Any, size: Tuple[int, int]) -> Image.Image:
    if image is None:
        return Image.new("L", size, 0)
    if isinstance(image, Image.Image):
        return image.convert("L").resize(size, Image.Resampling.NEAREST)
    if isinstance(image, bytes):
        return Image.open(io.BytesIO(image)).convert("L").resize(size, Image.Resampling.NEAREST)
    if isinstance(image, np.ndarray):
        return Image.fromarray(image).convert("L").resize(size, Image.Resampling.NEAREST)
    if isinstance(image, str):
        return Image.open(image).convert("L").resize(size, Image.Resampling.NEAREST)
    raise TypeError(f"Unsupported mask value: {type(image)!r}")


def _load_hf_dataset(name: str, split: str, cache_dir: Optional[str]):
    """Load HF datasets that still use dataset scripts, with one cache repair retry."""

    from datasets import load_dataset

    kwargs = {
        "split": split,
        "cache_dir": cache_dir,
        "trust_remote_code": True,
    }
    try:
        return load_dataset(name, **kwargs)
    except TypeError as exc:
        if "trust_remote_code" not in str(exc):
            raise
        kwargs.pop("trust_remote_code")
        try:
            return load_dataset(name, **kwargs)
        except UnicodeDecodeError as unicode_exc:
            return _retry_hf_dataset_after_decode_error(name, kwargs, unicode_exc)
    except UnicodeDecodeError as exc:
        return _retry_hf_dataset_after_decode_error(name, kwargs, exc)


def _retry_hf_dataset_after_decode_error(name: str, kwargs: Dict[str, Any], original_exc: UnicodeDecodeError):
    from datasets import DownloadMode, load_dataset

    retry_kwargs = dict(kwargs)
    retry_kwargs["download_mode"] = DownloadMode.FORCE_REDOWNLOAD
    try:
        return load_dataset(name, **retry_kwargs)
    except UnicodeDecodeError as exc:
        cache_hint = retry_kwargs.get("cache_dir") or "~/.cache/huggingface/datasets"
        raise RuntimeError(
            f"Failed to load Hugging Face dataset {name!r}: the local dataset module/cache appears to contain "
            "gzip or other binary data where the datasets loader expected UTF-8 text. Remove the dataset cache "
            f"under {cache_hint!r}, or materialize once with scripts/prepare_face_edit_data.py and train with "
            "--dataset_source canonical_jsonl --jsonl_path <prepared>/<split>.jsonl."
        ) from exc
    except TypeError as exc:
        if "trust_remote_code" not in str(exc):
            raise
        retry_kwargs.pop("trust_remote_code", None)
        try:
            return load_dataset(name, **retry_kwargs)
        except UnicodeDecodeError as exc:
            raise RuntimeError(
                f"Failed to load Hugging Face dataset {name!r} after refreshing the cache. Original error: "
                f"{original_exc}"
            ) from exc


def _local_wider_split_name(split: str) -> str:
    return "val" if split in {"validation", "val", "dev"} else "train"


def _find_zip_member(zip_file: zipfile.ZipFile, suffix: str) -> str:
    matches = [name for name in zip_file.namelist() if name.endswith(suffix)]
    if not matches:
        raise FileNotFoundError(f"Could not find {suffix!r} in {zip_file.filename!r}")
    return matches[0]


def _parse_wider_face_annotations(text: str) -> List[Dict[str, Any]]:
    lines = [line.strip() for line in text.splitlines() if line.strip()]
    records: List[Dict[str, Any]] = []
    idx = 0
    while idx < len(lines):
        image_relpath = lines[idx]
        idx += 1
        if idx >= len(lines):
            break
        face_count = int(lines[idx])
        idx += 1
        boxes: List[Tuple[float, float, float, float]] = []
        rows_to_read = max(face_count, 1)
        for row_idx in range(rows_to_read):
            if idx >= len(lines):
                break
            parts = lines[idx].split()
            idx += 1
            if row_idx < face_count and len(parts) >= 4:
                x, y, w, h = [float(value) for value in parts[:4]]
                if w > 1 and h > 1:
                    boxes.append((x, y, w, h))
        records.append({"image_relpath": image_relpath, "bboxes": boxes})
    return records


def _load_local_wider_face_records(root: str, split: str) -> List[Dict[str, Any]]:
    wider_split = _local_wider_split_name(split)
    root = os.path.abspath(root)
    image_zip_path = os.path.join(root, f"WIDER_{wider_split}.zip")
    annotation_zip_path = os.path.join(root, "wider_face_split.zip")
    if not os.path.exists(image_zip_path):
        raise FileNotFoundError(f"Missing WIDER image zip: {image_zip_path}")
    if not os.path.exists(annotation_zip_path):
        raise FileNotFoundError(f"Missing WIDER annotation zip: {annotation_zip_path}")

    annotation_name = f"wider_face_split/wider_face_{wider_split}_bbx_gt.txt"
    with zipfile.ZipFile(annotation_zip_path) as annotation_zip:
        annotation_member = _find_zip_member(annotation_zip, annotation_name)
        annotation_text = annotation_zip.read(annotation_member).decode("utf-8")

    records = _parse_wider_face_annotations(annotation_text)
    image_prefix = f"WIDER_{wider_split}/images/"
    return [
        {
            "image_zip_path": image_zip_path,
            "image_member": image_prefix + record["image_relpath"],
            "bboxes": record["bboxes"],
        }
        for record in records
    ]


def _load_image_dir_records(root: str) -> List[Dict[str, Any]]:
    root = os.path.abspath(root)
    if not os.path.isdir(root):
        raise FileNotFoundError(f"face_aug_preserve image_dir does not exist or is not a directory: {root}")

    paths: List[str] = []
    for ext in IMAGE_EXTENSIONS:
        paths.extend(glob(os.path.join(root, "**", f"*{ext}"), recursive=True))
        paths.extend(glob(os.path.join(root, "**", f"*{ext.upper()}"), recursive=True))
    paths = sorted(set(paths))
    if not paths:
        raise FileNotFoundError(f"No images with extensions {IMAGE_EXTENSIONS!r} found under {root}")
    return [{"image_path": path} for path in paths]


def _resize_triplet(
    source: Image.Image,
    target: Image.Image,
    mask: Image.Image,
    resolution: int,
) -> Tuple[Image.Image, Image.Image, Image.Image]:
    source = source.resize((resolution, resolution), Image.Resampling.BICUBIC)
    target = target.resize((resolution, resolution), Image.Resampling.BICUBIC)
    mask = mask.resize((resolution, resolution), Image.Resampling.NEAREST)
    return source, target, mask


def _to_tensor(image: Image.Image) -> torch.Tensor:
    return transforms.ToTensor()(image)


def _mask_to_tensor(mask: Image.Image) -> torch.Tensor:
    return transforms.ToTensor()(mask.convert("L")).clamp(0, 1)


def _extract_boxes(example: Dict[str, Any]) -> List[Tuple[float, float, float, float]]:
    """Best-effort parser for WIDER-FACE-like HF dataset records.

    Different mirrors expose boxes as `faces.bbox`, `annotations.bbox`, or a
    flat `bbox` list. Returned boxes are always xywh in image pixel space.
    """

    candidates = []
    if "faces" in example and isinstance(example["faces"], dict):
        candidates.extend([example["faces"].get("bbox"), example["faces"].get("bboxes")])
    if "annotations" in example and isinstance(example["annotations"], dict):
        candidates.extend([example["annotations"].get("bbox"), example["annotations"].get("bboxes")])
    candidates.extend([example.get("bbox"), example.get("bboxes"), example.get("face_bboxes")])

    boxes: List[Tuple[float, float, float, float]] = []
    for value in candidates:
        if value is None:
            continue
        arr = np.asarray(value, dtype=np.float32)
        if arr.size == 0:
            continue
        arr = arr.reshape(-1, arr.shape[-1])
        for row in arr:
            if len(row) < 4:
                continue
            x, y, w, h = [float(v) for v in row[:4]]
            if w > 1 and h > 1:
                boxes.append((x, y, w, h))
        if boxes:
            return boxes
    return boxes


def _box_mask(size: Tuple[int, int], boxes: Sequence[Tuple[float, float, float, float]], pad: float = 0.35) -> Image.Image:
    mask = Image.new("L", size, 0)
    if not boxes:
        return mask
    arr = np.array(mask)
    width, height = size
    for x, y, w, h in boxes:
        px = w * pad
        py = h * pad
        x0 = max(0, int(math.floor(x - px)))
        y0 = max(0, int(math.floor(y - py)))
        x1 = min(width, int(math.ceil(x + w + px)))
        y1 = min(height, int(math.ceil(y + h + py)))
        if x1 > x0 and y1 > y0:
            arr[y0:y1, x0:x1] = 255
    return Image.fromarray(arr, mode="L")


def _select_small_boxes(
    boxes: Sequence[Tuple[float, float, float, float]],
    image_size: Tuple[int, int],
    max_face_fraction: float,
    max_boxes: int,
) -> List[Tuple[float, float, float, float]]:
    width, height = image_size
    diag = math.sqrt(width * height)
    small = [box for box in boxes if max(box[2], box[3]) / max(diag, 1) <= max_face_fraction]
    chosen = small or list(boxes)
    chosen = sorted(chosen, key=lambda b: b[2] * b[3])
    return chosen[:max_boxes]


def _parse_task_mix(spec: str) -> List[Tuple[str, float]]:
    weights: List[Tuple[str, float]] = []
    for item in spec.split(","):
        item = item.strip()
        if not item:
            continue
        if ":" not in item:
            raise ValueError(f"Invalid task mix item {item!r}; expected name:weight")
        name, raw_weight = item.split(":", 1)
        name = name.strip()
        if name not in FACE_PRESERVE_PROMPTS:
            raise ValueError(f"Unsupported synthetic edit task {name!r}; expected one of {sorted(FACE_PRESERVE_PROMPTS)}")
        weight = float(raw_weight)
        if weight < 0:
            raise ValueError(f"Task mix weight must be non-negative, got {item!r}")
        if weight > 0:
            weights.append((name, weight))
    if not weights:
        raise ValueError(f"Task mix {spec!r} contains no positive weights")
    return weights


def _weighted_choice(rng: random.Random, weights: Sequence[Tuple[str, float]]) -> str:
    total = sum(weight for _, weight in weights)
    point = rng.random() * total
    upto = 0.0
    for name, weight in weights:
        upto += weight
        if point <= upto:
            return name
    return weights[-1][0]


def _choose_instruction(task: str, rng: random.Random, tag: str) -> str:
    instruction = rng.choice(FACE_PRESERVE_PROMPTS[task])
    if tag and tag not in instruction:
        instruction = f"{tag} {instruction}"
    return instruction


def _soft_face_mask(face_mask: Image.Image, blur_radius: float = 2.0) -> Image.Image:
    return face_mask.convert("L").filter(ImageFilter.GaussianBlur(blur_radius))


def synthetic_background_edit(image: Image.Image, face_mask: Image.Image, rng: random.Random) -> Image.Image:
    """Apply deterministic non-face edits, then paste original face regions back."""

    image = image.convert("RGB")
    op = rng.choice(["color", "exposure", "blur", "grayscale", "posterize", "noise", "tint"])
    if op == "color":
        edited = ImageEnhance.Color(image).enhance(rng.uniform(0.25, 1.9))
        edited = ImageEnhance.Contrast(edited).enhance(rng.uniform(0.75, 1.35))
    elif op == "exposure":
        edited = ImageEnhance.Brightness(image).enhance(rng.uniform(0.65, 1.35))
        edited = ImageEnhance.Contrast(edited).enhance(rng.uniform(0.8, 1.45))
    elif op == "blur":
        edited = image.filter(ImageFilter.GaussianBlur(rng.uniform(1.0, 3.0)))
    elif op == "grayscale":
        edited = ImageOps.grayscale(image).convert("RGB")
        edited = ImageEnhance.Contrast(edited).enhance(rng.uniform(0.8, 1.4))
    elif op == "posterize":
        edited = ImageOps.posterize(image, bits=rng.choice([3, 4, 5]))
    elif op == "noise":
        arr = np.asarray(image).astype(np.float32)
        np_rng = np.random.default_rng(rng.randrange(2**32))
        arr = np.clip(arr + np_rng.normal(0.0, rng.uniform(8.0, 24.0), size=arr.shape), 0, 255)
        edited = Image.fromarray(arr.astype(np.uint8), mode="RGB")
    else:
        color = tuple(rng.randrange(32, 224) for _ in range(3))
        overlay = Image.new("RGB", image.size, color)
        edited = Image.blend(image, overlay, alpha=rng.uniform(0.15, 0.35))

    result = edited.convert("RGB")
    result.paste(image, mask=_soft_face_mask(face_mask))
    return result


def degrade_face_regions(
    image: Image.Image,
    boxes: Sequence[Tuple[float, float, float, float]],
    downscale: int = 6,
    blur_radius: float = 0.8,
    jpeg_quality: int = 28,
) -> Image.Image:
    """Create low-resolution small-face inputs while keeping the full image aligned."""

    result = image.copy().convert("RGB")
    for x, y, w, h in boxes:
        x0 = max(0, int(x))
        y0 = max(0, int(y))
        x1 = min(result.width, int(x + w))
        y1 = min(result.height, int(y + h))
        if x1 <= x0 or y1 <= y0:
            continue
        crop = result.crop((x0, y0, x1, y1))
        low_w = max(2, crop.width // downscale)
        low_h = max(2, crop.height // downscale)
        crop = crop.resize((low_w, low_h), Image.Resampling.BICUBIC)
        crop = crop.resize((x1 - x0, y1 - y0), Image.Resampling.NEAREST)
        if blur_radius > 0:
            crop = crop.filter(ImageFilter.GaussianBlur(blur_radius))
        if jpeg_quality:
            buf = io.BytesIO()
            crop.save(buf, format="JPEG", quality=jpeg_quality)
            buf.seek(0)
            crop = Image.open(buf).convert("RGB")
        result.paste(crop, (x0, y0))
    return result


class FaceEditDataset(Dataset):
    """Unified dataloader for SFT and RL image-editing experiments.

    Supported `source` values:
    - `magicbrush`: real instruction-edit triples from `osunlp/MagicBrush`.
    - `wider_face_restore`: WIDER FACE images turned into low-res-face restoration pairs.
    - `face_aug_preserve`: local face images turned into synthetic preserve-ID edit pairs.
    - `canonical_jsonl`: materialized JSONL with canonical fields.

    The training contract is always:
    `source_image + instruction + mask_image -> target_image`.
    """

    def __init__(
        self,
        source: str,
        split: str = "train",
        resolution: int = 512,
        cache_dir: Optional[str] = None,
        jsonl_path: Optional[str] = None,
        image_dir: Optional[str] = None,
        wider_face_root: Optional[str] = None,
        max_samples: Optional[int] = None,
        seed: int = 42,
        small_face_fraction: float = 0.12,
        max_faces_per_image: int = 8,
        synthetic_task: str = "restore_lowres_small_face",
        synthetic_edit_mix: str = "restore:0.4,background:0.4,noop:0.2",
        face_detector_min_size: int = 8,
        face_prompt_tag: str = "[preserve face id]",
    ):
        self.source = source
        self.split = split
        self.resolution = resolution
        self.cache_dir = cache_dir
        self.max_samples = max_samples
        self.rng = random.Random(seed)
        self.small_face_fraction = small_face_fraction
        self.max_faces_per_image = max_faces_per_image
        self.synthetic_task = synthetic_task
        self.synthetic_edit_mix = _parse_task_mix(synthetic_edit_mix)
        self.face_detector_min_size = face_detector_min_size
        self.face_prompt_tag = face_prompt_tag
        self.records: Any
        self.jsonl_root = os.getcwd()

        if source == "canonical_jsonl":
            if not jsonl_path:
                raise ValueError("jsonl_path is required for canonical_jsonl")
            self.jsonl_root = os.path.dirname(os.path.abspath(jsonl_path))
            with open(jsonl_path, "r", encoding="utf-8") as f:
                self.records = [json.loads(line) for line in f if line.strip()]
        elif source == "magicbrush":
            hf_split = "dev" if split in {"validation", "val"} else split
            self.records = _load_hf_dataset("osunlp/MagicBrush", split=hf_split, cache_dir=cache_dir)
        elif source == "wider_face_restore":
            hf_split = "validation" if split in {"validation", "val", "dev"} else split
            if wider_face_root:
                self.records = _load_local_wider_face_records(wider_face_root, hf_split)
            else:
                self.records = _load_hf_dataset("CUHK-CSE/wider_face", split=hf_split, cache_dir=cache_dir)
        elif source == "face_aug_preserve":
            if not image_dir:
                raise ValueError("image_dir is required for face_aug_preserve")
            self.records = self._detect_face_image_records(_load_image_dir_records(image_dir))
        else:
            raise ValueError(f"Unsupported edit dataset source: {source}")

        if max_samples is not None:
            self.records = self.records.select(range(min(max_samples, len(self.records)))) if hasattr(self.records, "select") else self.records[:max_samples]

    def __len__(self) -> int:
        return len(self.records)

    def _detect_face_image_records(self, records: Sequence[Dict[str, Any]]) -> List[Dict[str, Any]]:
        filtered: List[Dict[str, Any]] = []
        for record in records:
            image = _pil_rgb(record["image_path"])
            boxes = detect_faces_opencv(image, min_size=self.face_detector_min_size)
            boxes = _select_small_boxes(boxes, image.size, self.small_face_fraction, self.max_faces_per_image)
            if boxes:
                filtered.append({**record, "bboxes": boxes})
        if not filtered:
            raise RuntimeError(
                "face_aug_preserve found no images with detectable faces. Check --image_dir, "
                "--face_detector_min_size, or install/use a stronger detector before materializing canonical_jsonl."
            )
        return filtered

    def _from_magicbrush(self, record: Dict[str, Any]) -> EditSample:
        source = _pil_rgb(record["source_img"])
        target = _pil_rgb(record["target_img"])
        mask = _pil_mask(record.get("mask_img"), source.size)
        if np.asarray(mask).max() == 0:
            mask = Image.new("L", source.size, 255)
        return EditSample(
            source_image=source,
            target_image=target,
            mask_image=mask,
            instruction=record["instruction"],
            metadata={"dataset": "magicbrush", "img_id": record.get("img_id"), "turn_index": record.get("turn_index")},
        )

    def _from_wider_face(self, record: Dict[str, Any]) -> EditSample:
        if "image_zip_path" in record:
            with zipfile.ZipFile(record["image_zip_path"]) as image_zip:
                image = _pil_rgb(image_zip.read(record["image_member"]))
        else:
            image = _pil_rgb(record.get("image") or record.get("img"))
        boxes = _extract_boxes(record)
        boxes = _select_small_boxes(boxes, image.size, self.small_face_fraction, self.max_faces_per_image)
        if not boxes:
            width, height = image.size
            side = min(width, height) * 0.12
            boxes = [(width * 0.44, height * 0.28, side, side)]
        source = degrade_face_regions(image, boxes)
        mask = _box_mask(image.size, boxes)
        instruction = "restore the small low-resolution face while preserving the rest of the image"
        return EditSample(
            source_image=source,
            target_image=image,
            mask_image=mask,
            instruction=instruction,
            metadata={"dataset": "wider_face", "boxes_xywh": boxes, "synthetic_task": self.synthetic_task},
        )

    def _from_face_aug_preserve(self, record: Dict[str, Any], idx: int) -> EditSample:
        image = _pil_rgb(record["image_path"])
        boxes = record["bboxes"]
        face_mask = _box_mask(image.size, boxes)
        rng = random.Random(self.seed + idx * 1009)
        task = _weighted_choice(rng, self.synthetic_edit_mix)

        if task == "restore":
            source = degrade_face_regions(image, boxes)
            target = image
        elif task == "background":
            source = image
            target = synthetic_background_edit(image, face_mask, rng)
        elif task == "noop":
            source = image
            target = image
        else:
            raise AssertionError(task)

        return EditSample(
            source_image=source,
            target_image=target,
            mask_image=face_mask,
            instruction=_choose_instruction(task, rng, self.face_prompt_tag),
            metadata={
                "dataset": "face_aug_preserve",
                "image_path": record["image_path"],
                "boxes_xywh": boxes,
                "synthetic_task": task,
                "face_prompt_tag": self.face_prompt_tag,
            },
        )

    def _from_jsonl(self, record: Dict[str, Any]) -> EditSample:
        def resolve(path: str) -> str:
            return path if os.path.isabs(path) else os.path.join(self.jsonl_root, path)

        source = _pil_rgb(resolve(record["source_image"]))
        target = _pil_rgb(resolve(record["target_image"]))
        mask = _pil_mask(resolve(record["mask_image"]), source.size) if record.get("mask_image") else Image.new("L", source.size, 255)
        return EditSample(source, target, mask, record["instruction"], record.get("metadata", {}))

    def get_sample(self, idx: int) -> EditSample:
        record = self.records[idx]
        if self.source == "magicbrush":
            sample = self._from_magicbrush(record)
        elif self.source == "wider_face_restore":
            sample = self._from_wider_face(record)
        elif self.source == "face_aug_preserve":
            sample = self._from_face_aug_preserve(record, idx)
        elif self.source == "canonical_jsonl":
            sample = self._from_jsonl(record)
        else:
            raise AssertionError(self.source)
        sample.source_image, sample.target_image, sample.mask_image = _resize_triplet(
            sample.source_image, sample.target_image, sample.mask_image, self.resolution
        )
        return sample

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        sample = self.get_sample(idx)
        return {
            "source_images": _to_tensor(sample.source_image),
            "target_images": _to_tensor(sample.target_image),
            "masks": _mask_to_tensor(sample.mask_image),
            "instructions": sample.instruction,
            "metadata": sample.metadata,
        }


def collate_edit_samples(examples: Sequence[Dict[str, Any]]) -> Dict[str, Any]:
    return {
        "source_images": torch.stack([ex["source_images"] for ex in examples]),
        "target_images": torch.stack([ex["target_images"] for ex in examples]),
        "masks": torch.stack([ex["masks"] for ex in examples]),
        "instructions": [ex["instructions"] for ex in examples],
        "metadata": [ex["metadata"] for ex in examples],
    }
