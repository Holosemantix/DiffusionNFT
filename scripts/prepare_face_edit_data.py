"""Download/materialize face-edit datasets into the canonical JSONL format.

Examples:
  python scripts/prepare_face_edit_data.py --source wider_face_restore --split train --output_dir data/face_edit/wider_train --max_samples 2000
  python scripts/prepare_face_edit_data.py --source wider_face_restore --wider_face_root /path/to/wider_face_zips --split train --output_dir data/face_edit/wider_train --max_samples 2000
  python scripts/prepare_face_edit_data.py --source magicbrush --split train --output_dir data/face_edit/magicbrush_train --max_samples 2000
  python scripts/prepare_face_edit_data.py --source face_aug_preserve --image_dir /path/to/face_images_or_zips --output_dir data/face_edit/face_aug_train
"""

import argparse
import json
import os

from tqdm import tqdm

from flow_grpo.editing_data import CANONICAL_FIELDS, FaceEditDataset


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", default="wider_face_restore", choices=["wider_face_restore", "magicbrush", "face_aug_preserve"])
    parser.add_argument("--split", default="train")
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--cache_dir", default=None)
    parser.add_argument("--wider_face_root", default=None, help="Directory containing WIDER_train.zip/WIDER_val.zip and wider_face_split.zip")
    parser.add_argument("--image_dir", default=None, help="Directory of ordinary face images for face_aug_preserve.")
    parser.add_argument(
        "--zip_extract_dir",
        default=None,
        help="Optional directory for losslessly extracted zip archives under --image_dir. Defaults to <image_dir>/_unzipped.",
    )
    parser.add_argument("--resolution", type=int, default=512)
    parser.add_argument("--max_samples", type=int, default=None)
    parser.add_argument(
        "--synthetic_edit_mix",
        default="restore:0.4,background:0.4,noop:0.2",
        help="Task weights for face_aug_preserve, e.g. restore:0.4,background:0.4,noop:0.2.",
    )
    parser.add_argument("--face_detector_min_size", type=int, default=8,
                        help="Minimum OpenCV-detected face size for face_aug_preserve filtering.")
    parser.add_argument("--face_prompt_tag", default="[preserve face id]",
                        help="Trigger tag prepended to preserve-ID synthetic instructions.")
    return parser.parse_args()


def main():
    args = parse_args()
    os.makedirs(args.output_dir, exist_ok=True)
    image_dir = os.path.join(args.output_dir, "images")
    mask_dir = os.path.join(args.output_dir, "masks")
    os.makedirs(image_dir, exist_ok=True)
    os.makedirs(mask_dir, exist_ok=True)

    dataset = FaceEditDataset(
        source=args.source,
        split=args.split,
        resolution=args.resolution,
        cache_dir=args.cache_dir,
        wider_face_root=args.wider_face_root,
        image_dir=args.image_dir,
        zip_extract_dir=args.zip_extract_dir,
        max_samples=args.max_samples,
        synthetic_edit_mix=args.synthetic_edit_mix,
        face_detector_min_size=args.face_detector_min_size,
        face_prompt_tag=args.face_prompt_tag,
    )

    jsonl_path = os.path.join(args.output_dir, f"{args.split}.jsonl")
    schema_path = os.path.join(args.output_dir, "schema.json")
    with open(schema_path, "w", encoding="utf-8") as f:
        json.dump(CANONICAL_FIELDS, f, indent=2)

    with open(jsonl_path, "w", encoding="utf-8") as f:
        for idx in tqdm(range(len(dataset)), desc=f"materializing {args.source}:{args.split}"):
            sample = dataset.get_sample(idx)
            source_rel = f"images/{idx:07d}_source.jpg"
            target_rel = f"images/{idx:07d}_target.jpg"
            mask_rel = f"masks/{idx:07d}_mask.png"
            sample.source_image.save(os.path.join(args.output_dir, source_rel), quality=95)
            sample.target_image.save(os.path.join(args.output_dir, target_rel), quality=95)
            sample.mask_image.save(os.path.join(args.output_dir, mask_rel))
            f.write(
                json.dumps(
                    {
                        "source_image": source_rel,
                        "target_image": target_rel,
                        "mask_image": mask_rel,
                        "instruction": sample.instruction,
                        "metadata": sample.metadata,
                    },
                    ensure_ascii=True,
                )
                + "\n"
            )
    print(f"Wrote {len(dataset)} samples to {jsonl_path}")


if __name__ == "__main__":
    main()
