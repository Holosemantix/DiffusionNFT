import io
import json
import math
import os
import random
from dataclasses import dataclass
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import torch
from PIL import Image, ImageFilter
from torch.utils.data import Dataset
from torchvision import transforms


CANONICAL_FIELDS = {
    "source_image": "PIL RGB image before editing",
    "target_image": "PIL RGB image after editing or restoration",
    "mask_image": "PIL L image, white means editable/evaluated region",
    "instruction": "text instruction for image editing",
    "metadata": "dataset-specific metadata",
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
        max_samples: Optional[int] = None,
        seed: int = 42,
        small_face_fraction: float = 0.12,
        max_faces_per_image: int = 8,
        synthetic_task: str = "restore_lowres_small_face",
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
        self.records: Any
        self.jsonl_root = os.getcwd()

        if source == "canonical_jsonl":
            if not jsonl_path:
                raise ValueError("jsonl_path is required for canonical_jsonl")
            self.jsonl_root = os.path.dirname(os.path.abspath(jsonl_path))
            with open(jsonl_path, "r", encoding="utf-8") as f:
                self.records = [json.loads(line) for line in f if line.strip()]
        elif source == "magicbrush":
            from datasets import load_dataset

            hf_split = "dev" if split in {"validation", "val"} else split
            self.records = load_dataset("osunlp/MagicBrush", split=hf_split, cache_dir=cache_dir)
        elif source == "wider_face_restore":
            from datasets import load_dataset

            hf_split = "validation" if split in {"validation", "val", "dev"} else split
            self.records = load_dataset("CUHK-CSE/wider_face", split=hf_split, cache_dir=cache_dir)
        else:
            raise ValueError(f"Unsupported edit dataset source: {source}")

        if max_samples is not None:
            self.records = self.records.select(range(min(max_samples, len(self.records)))) if hasattr(self.records, "select") else self.records[:max_samples]

    def __len__(self) -> int:
        return len(self.records)

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
