from pathlib import Path
from typing import Optional, Dict, Any
import torch


def read_checkpoint_metrics(models_dir: Path) -> Dict[str, Any]:
    """
    Read training metrics from the best model checkpoint if available.

    Args:
        models_dir: Base models directory containing best_model.pt

    Returns:
        dict with keys: model_exists (bool), val_acc (float|None), val_loss (float|None),
        epoch (int|None), model_path (str|None)
    """
    ckpt_path = Path(models_dir) / "best_model.pt"
    result: Dict[str, Any] = {
        "model_exists": ckpt_path.exists(),
        "val_acc": None,
        "val_loss": None,
        "epoch": None,
        "model_path": str(ckpt_path) if ckpt_path.exists() else None,
    }

    if not ckpt_path.exists():
        return result

    try:
        checkpoint = torch.load(ckpt_path, map_location="cpu")
        result["val_acc"] = float(checkpoint.get("val_acc")) if checkpoint.get("val_acc") is not None else None
        result["val_loss"] = float(checkpoint.get("val_loss")) if checkpoint.get("val_loss") is not None else None
        result["epoch"] = int(checkpoint.get("epoch")) if checkpoint.get("epoch") is not None else None
    except Exception:
        # Return what we have; metrics remain None on failure
        pass

    return result
