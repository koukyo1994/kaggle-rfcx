import torch

from pathlib import Path


def save_model(model, logdir: Path, filename: str):
    state_dict = {}
    state_dict["model_state_dict"] = model.state_dict()

    weights_path = logdir / filename
    with open(weights_path, "wb") as f:
        torch.save(state_dict, f)


def save_best_model(model, logdir, metric: float, prev_metric: float):
    if metric > prev_metric:
        save_model(model, logdir, "best.pth")
        return metric, True
    else:
        return prev_metric, False
