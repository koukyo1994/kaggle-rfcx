import torch
import torch.nn as nn
import torch.optim as optim

from iterstrat.ml_stratifiers import MultilabelStratifiedKFold
from sklearn import model_selection

from .optimizers import AdaBelief


__OPTIMIZERS__ = {
    "AdaBelief": AdaBelief
}


def get_device(device: str):
    if torch.cuda.is_available() and "cuda" in device:
        return torch.device(device)
    else:
        return torch.device("cpu")


def get_optimizer(model: nn.Module, config: dict):
    optimizer_config = config["optimizer"]
    optimizer_name = optimizer_config.get("name")
    if __OPTIMIZERS__.get(optimizer_name) is not None:
        return __OPTIMIZERS__[optimizer_name](model.parameters(),
                                              **optimizer_config["params"])
    else:
        return optim.__getattribute__(optimizer_name)(model.parameters(),
                                                      **optimizer_config["params"])


def get_scheduler(optimizer, config: dict):
    scheduler_config = config["scheduler"]
    scheduler_name = scheduler_config.get("name")

    if scheduler_name is None:
        return
    else:
        return optim.lr_scheduler.__getattribute__(scheduler_name)(
            optimizer, **scheduler_config["params"])


def get_split(config: dict):
    split_config = config["split"]
    name = split_config["name"]

    if hasattr(model_selection, name):
        return model_selection.__getattribute__(name)(**split_config["params"])
    else:
        return MultilabelStratifiedKFold(**split_config["params"])
