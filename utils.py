import json
import os
import random
import time
from typing import Any, Dict

import numpy as np

try:
    import torch
except Exception:  # pragma: no cover - torch optional for non-ML scripts
    torch = None


def seed_all(seed: int, deterministic: bool = True) -> None:
    """Seed python, numpy, and torch (if available) for reproducibility."""
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    if torch is None:
        return
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    if deterministic:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def now_id() -> str:
    return time.strftime("%Y%m%d_%H%M%S")


def save_json(path: str, data: Dict[str, Any]) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)
