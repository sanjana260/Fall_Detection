import os
import urllib.request
from typing import Dict, Any, List, Optional
import torch
import torch.nn.functional as F
import torchvision.transforms as T
import torchvision.models as models
from PIL import Image


# --- Remote assets (official Places365 resources) ---
PLACES_MODEL_URL = "http://places2.csail.mit.edu/models_places365/resnet18_places365.pth.tar"
CATEGORIES_URL = "https://raw.githubusercontent.com/csailvision/places365/master/categories_places365.txt"
IO_LABELS_URL = "https://raw.githubusercontent.com/csailvision/places365/master/IO_places365.txt"


def _cache_dir() -> str:
    return os.path.join(os.path.expanduser("~"), ".cache", "places365")


def _download(url: str, path: str) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    if os.path.exists(path) and os.path.getsize(path) > 0:
        return
    print(f"[SceneContext] Downloading {url} -> {path}")
    urllib.request.urlretrieve(url, path)


def _load_categories(path: str) -> List[str]:
    """
    categories_places365.txt formats vary. Common formats:
    1) '0 /a/airfield'
    2) '000 /a/airfield 0'
    3) '/a/airfield'
    We want a clean list of 365 class names like 'airfield'.
    """
    cats: List[str] = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            s = line.strip()
            if not s:
                continue
            parts = s.split()

            # Find the token that looks like a Places category: starts with '/'
            cat_token = None
            for tok in parts:
                if tok.startswith("/"):
                    cat_token = tok
                    break

            # If we didn't find '/something', fallback to last token
            if cat_token is None:
                cat_token = parts[-1]

            # Remove '/x/' prefix if present (e.g. '/a/airfield' -> 'airfield')
            if len(cat_token) >= 3 and cat_token[0] == "/" and cat_token[2] == "/":
                cat_token = cat_token[3:]

            cats.append(cat_token)

    return cats



def _load_io_labels(path: str) -> List[int]:
    labels: List[int] = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            s = line.strip()
            if not s:
                continue

            parts = s.split()
            last = parts[-1]

            try:
                v = int(last)
            except ValueError:
                continue

            # Normalize to: indoor=-1, outdoor=1, ambiguous=0
            # Some files use 0=ambiguous, 1=indoor, 2=outdoor
            if v == 2:
                labels.append(1)     # outdoor
            elif v == 1:
                labels.append(-1)    # indoor
            else:
                labels.append(0)     # ambiguous
    return labels


def _build_model(device: torch.device) -> torch.nn.Module:
    # Places365 uses 365 output classes
    model = models.resnet18(num_classes=365)

    # Download checkpoint via torch hub helper (respects cache dir)
    ckpt = torch.hub.load_state_dict_from_url(
        PLACES_MODEL_URL, model_dir=_cache_dir(), map_location=device
    )
    state_dict = ckpt.get("state_dict", ckpt)

    cleaned = {}
    for k, v in state_dict.items():
        cleaned[k.replace("module.", "")] = v

    model.load_state_dict(cleaned, strict=True)
    model.to(device)
    model.eval()
    return model


class SceneContext:
    """
    Places365 scene context with temporal averaging.

    Two outputs:
    1) realtime (EMA-smoothed) prediction per frame
    2) final prediction for the whole video (true average of all frames seen)
    """

    def __init__(
        self,
        device: str = "auto",
        ema_alpha: float = 0.30,          # 0.2-0.4 good; lower = smoother
        conf_threshold: float = 0.35      # below this: output 'uncertain_scene'
    ):
        self.device = torch.device(
            "cuda" if (device == "auto" and torch.cuda.is_available()) else ("cpu" if device == "auto" else device)
        )

        cache = _cache_dir()
        self.model_path = os.path.join(cache, "resnet18_places365.pth.tar")
        self.categories_path = os.path.join(cache, "categories_places365.txt")
        self.io_path = os.path.join(cache, "IO_places365.txt")

        _download(CATEGORIES_URL, self.categories_path)
        _download(IO_LABELS_URL, self.io_path)

        self.categories = _load_categories(self.categories_path)
        self.io_labels = _load_io_labels(self.io_path)

        if len(self.categories) != 365:
            print(f"[SceneContext] Warning: categories count = {len(self.categories)} (expected 365)")
        if len(self.io_labels) != 365:
            print(f"[SceneContext] Warning: IO labels count = {len(self.io_labels)} (expected 365)")

        self.model = _build_model(self.device)

        self.tf = T.Compose([
            T.Resize((256, 256)),
            T.CenterCrop(224),
            T.ToTensor(),
            T.Normalize(mean=[0.485, 0.456, 0.406],
                        std=[0.229, 0.224, 0.225]),
        ])

        self.ema_alpha = float(ema_alpha)
        self.conf_threshold = float(conf_threshold)

        # Realtime smoothing state (EMA of probabilities)
        self._ema_probs: Optional[torch.Tensor] = None  # CPU tensor

        # Whole-video averaging state
        self._sum_probs: Optional[torch.Tensor] = None  # CPU tensor
        self._count: int = 0

    def reset(self) -> None:
        """Call when starting a new video."""
        self._ema_probs = None
        self._sum_probs = None
        self._count = 0

    @torch.inference_mode()
    def _forward_probs(self, img: Image.Image) -> torch.Tensor:
        x = self.tf(img.convert("RGB")).unsqueeze(0).to(self.device)
        logits = self.model(x)
        probs = F.softmax(logits, dim=1).squeeze(0).detach().cpu()
        return probs  # shape [365] on CPU

    def _update_ema(self, probs: torch.Tensor) -> torch.Tensor:
        if self._ema_probs is None:
            self._ema_probs = probs
        else:
            a = self.ema_alpha
            self._ema_probs = a * probs + (1.0 - a) * self._ema_probs
        return self._ema_probs

    def _update_global_avg(self, probs: torch.Tensor) -> None:
        if self._sum_probs is None:
            self._sum_probs = probs.clone()
            self._count = 1
        else:
            self._sum_probs += probs
            self._count += 1

    def _topk_from_probs(self, probs: torch.Tensor, topk: int):
        top_probs, top_idxs = torch.topk(probs, k=min(topk, probs.shape[0]))
        top_probs = top_probs.tolist()
        top_idxs = top_idxs.tolist()
        top_labels = [self.categories[i] if i < len(self.categories) else f"class_{i}" for i in top_idxs]
        return top_labels, top_probs, top_idxs

    def _indoor_outdoor_vote(self, probs: torch.Tensor, topk: int) -> (str, float):
        labels, ps, idxs = self._topk_from_probs(probs, topk=topk)

        weighted = 0.0
        weight_sum = 0.0
        for idx, p in zip(idxs, ps):
            io = self.io_labels[idx] if idx < len(self.io_labels) else 0
            if io != 0:
                weighted += (1.0 if io == 1 else -1.0) * p
                weight_sum += p

        if weight_sum == 0:
            return "unknown", 0.0

        io = "outdoor" if weighted > 0 else "indoor"
        io_conf = abs(weighted) / weight_sum
        return io, float(io_conf)

    @torch.inference_mode()
    def predict_bgr(self, frame_bgr, topk: int = 5) -> Dict[str, Any]:
        """
        Call this per frame. Returns EMA-smoothed realtime prediction.
        Also updates whole-video average stats internally.
        """
        img = Image.fromarray(frame_bgr[:, :, ::-1])  # BGR -> RGB
        probs = self._forward_probs(img)

        # update smoothing + global average
        ema_probs = self._update_ema(probs)
        self._update_global_avg(probs)

        top_labels, top_probs, _ = self._topk_from_probs(ema_probs, topk=topk)
        scene_raw = top_labels[0]
        scene_conf = float(top_probs[0])

        # only claim a specific scene if confidence is decent
        scene = scene_raw if scene_conf >= self.conf_threshold else "uncertain_scene"

        io, io_conf = self._indoor_outdoor_vote(ema_probs, topk=topk)

        return {
            "scene_type_raw": scene_raw,
            "scene_conf_raw": scene_conf,
            "scene_type": scene,
            "scene_conf": scene_conf if scene != "uncertain_scene" else 0.0,
            "topk": [{"label": l, "prob": float(p)} for l, p in zip(top_labels, top_probs)],
            "indoor_outdoor": io,
            "io_conf": io_conf,
            "device": str(self.device),
            "smoothed": True,
            "ema_alpha": self.ema_alpha,
            "conf_threshold": self.conf_threshold,
            "frames_seen": self._count,
        }

    def final_video_prediction(self, topk: int = 5) -> Dict[str, Any]:
        """
        Call this once at the end of the video to get a single final scene prediction
        based on true average over all frames processed.
        """
        if self._sum_probs is None or self._count == 0:
            return {
                "scene_type": "unknown",
                "scene_conf": 0.0,
                "indoor_outdoor": "unknown",
                "io_conf": 0.0,
                "frames_seen": 0,
            }

        avg_probs = self._sum_probs / float(self._count)
        top_labels, top_probs, _ = self._topk_from_probs(avg_probs, topk=topk)

        scene_raw = top_labels[0]
        scene_conf = float(top_probs[0])
        scene = scene_raw if scene_conf >= self.conf_threshold else "uncertain_scene"

        io, io_conf = self._indoor_outdoor_vote(avg_probs, topk=topk)

        return {
            "scene_type_raw": scene_raw,
            "scene_conf_raw": scene_conf,
            "scene_type": scene,
            "scene_conf": scene_conf if scene != "uncertain_scene" else 0.0,
            "topk": [{"label": l, "prob": float(p)} for l, p in zip(top_labels, top_probs)],
            "indoor_outdoor": io,
            "io_conf": io_conf,
            "frames_seen": self._count,
            "smoothed": False,
            "final_average": True,
        }
