import math
from functools import lru_cache
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from torchvision import transforms


def charbonnier_loss(pred: torch.Tensor, target: torch.Tensor, mask: Optional[torch.Tensor] = None, eps: float = 1e-3):
    loss = torch.sqrt((pred - target) ** 2 + eps**2)
    if mask is not None:
        while mask.ndim < loss.ndim:
            mask = mask.unsqueeze(1)
        loss = loss * mask
        return loss.sum() / mask.sum().clamp_min(1.0)
    return loss.mean()


def downsample_mask(mask: torch.Tensor, size: Tuple[int, int]) -> torch.Tensor:
    return F.interpolate(mask.float(), size=size, mode="nearest").clamp(0, 1)


def multiscale_charbonnier(
    pred: torch.Tensor,
    target: torch.Tensor,
    mask: Optional[torch.Tensor] = None,
    scales: Sequence[int] = (1, 2, 4),
) -> torch.Tensor:
    losses = []
    for scale in scales:
        if scale == 1:
            p, t = pred, target
            m = mask
        else:
            h, w = pred.shape[-2] // scale, pred.shape[-1] // scale
            p = F.interpolate(pred, size=(h, w), mode="bilinear", align_corners=False)
            t = F.interpolate(target, size=(h, w), mode="bilinear", align_corners=False)
            m = downsample_mask(mask, (h, w)) if mask is not None else None
        losses.append(charbonnier_loss(p, t, m))
    return torch.stack(losses).mean()


def robust_edit_supervision_loss(
    pred_target: torch.Tensor,
    target: torch.Tensor,
    source: torch.Tensor,
    mask: torch.Tensor,
    edit_weight: float = 1.0,
    preserve_weight: float = 0.25,
    global_weight: float = 0.1,
) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
    """Mask-aware SFT loss that tolerates imperfect before/after alignment.

    The editable face region uses a robust multiscale target loss. The
    background uses a weaker preserve loss against the source image, which is
    more reliable when the target is not pixel-perfectly aligned with source.
    """

    mask = mask.float().clamp(0, 1)
    inv_mask = 1.0 - mask
    edit_loss = multiscale_charbonnier(pred_target, target, mask)
    preserve_loss = multiscale_charbonnier(pred_target, source, inv_mask, scales=(2, 4))
    global_loss = charbonnier_loss(pred_target, target)
    total = edit_weight * edit_loss + preserve_weight * preserve_loss + global_weight * global_loss
    return total, {
        "sft_edit_loss": edit_loss.detach(),
        "sft_preserve_loss": preserve_loss.detach(),
        "sft_global_loss": global_loss.detach(),
    }


def _pil_to_tensor(image: Image.Image) -> torch.Tensor:
    return transforms.ToTensor()(image.convert("RGB"))


def _tensor_to_pil(image: torch.Tensor) -> Image.Image:
    image = image.detach().float().cpu().clamp(0, 1)
    arr = (image.permute(1, 2, 0).numpy() * 255).round().astype(np.uint8)
    return Image.fromarray(arr)


def _mask_bbox(mask: torch.Tensor, pad: int = 8) -> Optional[Tuple[int, int, int, int]]:
    if mask.ndim == 3:
        mask = mask[0]
    ys, xs = torch.where(mask.detach().cpu() > 0.2)
    if len(xs) == 0:
        return None
    x0 = max(0, int(xs.min()) - pad)
    y0 = max(0, int(ys.min()) - pad)
    x1 = min(mask.shape[-1], int(xs.max()) + pad + 1)
    y1 = min(mask.shape[-2], int(ys.max()) + pad + 1)
    return x0, y0, x1, y1


def _crop_tensor(image: torch.Tensor, bbox: Optional[Tuple[int, int, int, int]]) -> torch.Tensor:
    if bbox is None:
        return image
    x0, y0, x1, y1 = bbox
    return image[..., y0:y1, x0:x1]


def _cosine_histogram(a: torch.Tensor, b: torch.Tensor, bins: int = 32) -> float:
    # Dependency-free identity proxy used when facenet-pytorch is unavailable.
    hists = []
    for image in [a, b]:
        vals = image.detach().float().cpu().clamp(0, 1)
        per_channel = [torch.histc(vals[c], bins=bins, min=0, max=1) for c in range(vals.shape[0])]
        hist = torch.cat(per_channel)
        hist = hist / hist.norm().clamp_min(1e-6)
        hists.append(hist)
    return float(torch.dot(hists[0], hists[1]).clamp(0, 1))


@lru_cache(maxsize=1)
def _opencv_face_detector():
    try:
        import cv2

        path = cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
        detector = cv2.CascadeClassifier(path)
        return cv2, detector
    except Exception:
        return None, None


def detect_faces_opencv(image: Image.Image, min_size: int = 8) -> List[Tuple[int, int, int, int]]:
    cv2, detector = _opencv_face_detector()
    if cv2 is None or detector is None:
        return []
    arr = np.asarray(image.convert("RGB"))
    gray = cv2.cvtColor(arr, cv2.COLOR_RGB2GRAY)
    faces = detector.detectMultiScale(gray, scaleFactor=1.05, minNeighbors=3, minSize=(min_size, min_size))
    return [(int(x), int(y), int(w), int(h)) for x, y, w, h in faces]


class OptionalFaceEmbedder:
    def __init__(self, device: torch.device):
        self.device = device
        self.model = None
        try:
            from facenet_pytorch import InceptionResnetV1

            self.model = InceptionResnetV1(pretrained="vggface2").eval().to(device)
        except Exception:
            self.model = None

    @torch.no_grad()
    def similarity(self, a: torch.Tensor, b: torch.Tensor) -> float:
        if self.model is None:
            return _cosine_histogram(a, b)
        a = F.interpolate(a.unsqueeze(0).to(self.device), size=(160, 160), mode="bilinear", align_corners=False)
        b = F.interpolate(b.unsqueeze(0).to(self.device), size=(160, 160), mode="bilinear", align_corners=False)
        a = (a - 0.5) / 0.5
        b = (b - 0.5) / 0.5
        ea = F.normalize(self.model(a), dim=-1)
        eb = F.normalize(self.model(b), dim=-1)
        return float((ea * eb).sum(dim=-1).clamp(-1, 1).add(1).mul(0.5).item())


class OptionalLandmarkScorer:
    def __init__(self):
        self.mp_face_mesh = None
        self.face_mesh = None
        try:
            import mediapipe as mp

            self.mp_face_mesh = mp.solutions.face_mesh
            self.face_mesh = self.mp_face_mesh.FaceMesh(static_image_mode=True, max_num_faces=1, refine_landmarks=True)
        except Exception:
            self.mp_face_mesh = None
            self.face_mesh = None

    def _landmarks(self, image: Image.Image) -> Optional[np.ndarray]:
        if self.face_mesh is None:
            return None
        arr = np.asarray(image.convert("RGB"))
        result = self.face_mesh.process(arr)
        if not result.multi_face_landmarks:
            return None
        pts = [(lm.x, lm.y, lm.z) for lm in result.multi_face_landmarks[0].landmark]
        return np.asarray(pts, dtype=np.float32)

    def similarity(self, a: Image.Image, b: Image.Image) -> float:
        la = self._landmarks(a)
        lb = self._landmarks(b)
        if la is None or lb is None:
            # Fallback: compare low-frequency grayscale structure.
            ta = _pil_to_tensor(a.convert("L").resize((32, 32))).flatten()
            tb = _pil_to_tensor(b.convert("L").resize((32, 32))).flatten()
            return float(F.cosine_similarity(ta, tb, dim=0).clamp(-1, 1).add(1).mul(0.5).item())
        distance = np.linalg.norm(la[:, :2] - lb[:, :2], axis=1).mean()
        return float(max(0.0, 1.0 - distance * 12.0))


class FaceEditRewardScorer:
    """Reward model for small-face editing RL.

    Detection, identity, landmark, and non-edit preservation are all included
    in the scalar reward. Optional heavy face models are used when installed;
    otherwise deterministic lightweight proxies keep the training code usable.
    """

    def __init__(
        self,
        device: torch.device,
        detection_weight: float = 1.0,
        identity_weight: float = 1.0,
        landmark_weight: float = 0.5,
        preserve_weight: float = 0.75,
        target_weight: float = 0.5,
    ):
        self.device = device
        self.weights = {
            "face_detection": detection_weight,
            "identity": identity_weight,
            "landmark": landmark_weight,
            "preserve": preserve_weight,
            "target": target_weight,
        }
        self.embedder = OptionalFaceEmbedder(device)
        self.landmarks = OptionalLandmarkScorer()

    def score_batch(
        self,
        generated: torch.Tensor,
        source: torch.Tensor,
        target: Optional[torch.Tensor],
        masks: torch.Tensor,
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        scores: List[float] = []
        details: Dict[str, List[float]] = {key: [] for key in self.weights}

        for idx in range(generated.shape[0]):
            gen = generated[idx].detach().float().cpu().clamp(0, 1)
            src = source[idx].detach().float().cpu().clamp(0, 1)
            tgt = target[idx].detach().float().cpu().clamp(0, 1) if target is not None else None
            mask = masks[idx].detach().float().cpu().clamp(0, 1)
            bbox = _mask_bbox(mask)
            gen_crop = _crop_tensor(gen, bbox)
            src_crop = _crop_tensor(src, bbox)
            tgt_crop = _crop_tensor(tgt, bbox) if tgt is not None else None

            gen_pil = _tensor_to_pil(gen_crop)
            tgt_or_src_pil = _tensor_to_pil(tgt_crop if tgt_crop is not None else src_crop)

            detected = 1.0 if detect_faces_opencv(gen_pil, min_size=6) else 0.0
            identity = self.embedder.similarity(gen_crop, tgt_crop if tgt_crop is not None else src_crop)
            landmark = self.landmarks.similarity(gen_pil, tgt_or_src_pil)
            inv_mask = (1.0 - mask).expand_as(gen)
            preserve_l1 = (torch.abs(gen - src) * inv_mask).sum() / inv_mask.sum().clamp_min(1.0)
            preserve = float((1.0 - preserve_l1 * 4.0).clamp(0, 1).item())
            if tgt_crop is not None:
                target_l1 = torch.abs(gen_crop - tgt_crop).mean()
                target_score = float((1.0 - target_l1 * 4.0).clamp(0, 1).item())
            else:
                target_score = 0.0

            details["face_detection"].append(detected)
            details["identity"].append(identity)
            details["landmark"].append(landmark)
            details["preserve"].append(preserve)
            details["target"].append(target_score)

            total_weight = sum(abs(v) for v in self.weights.values())
            score = sum(self.weights[k] * details[k][-1] for k in self.weights) / max(total_weight, 1e-6)
            scores.append(score)

        score_tensor = torch.tensor(scores, device=self.device, dtype=torch.float32)
        detail_tensors = {k: torch.tensor(v, device=self.device, dtype=torch.float32) for k, v in details.items()}
        return score_tensor, detail_tensors

