import torch
import torch.nn as nn
import torch.optim as optim

from catalyst.dl import SupervisedRunner
from iterstrat.ml_stratifiers import MultilabelStratifiedKFold
from sklearn import model_selection

from .optimizers import AdaBelief, SAM
from .runners import SAMRunner


__OPTIMIZERS__ = {
    "AdaBelief": AdaBelief,
    "SAM": SAM
}


def get_device(device: str):
    if torch.cuda.is_available() and "cuda" in device:
        return torch.device(device)
    else:
        return torch.device("cpu")


def get_optimizer(model: nn.Module, config: dict):
    optimizer_config = config["optimizer"]
    optimizer_name = optimizer_config.get("name")
    if optimizer_name == "SAM":
        base_optimizer_name = optimizer_config.get("base_optimizer")
        if __OPTIMIZERS__.get(base_optimizer_name) is not None:
            base_optimizer = __OPTIMIZERS__[base_optimizer_name]
        else:
            base_optimizer = optim.__getattribute__(base_optimizer_name)
        return SAM(model.parameters(), base_optimizer, **optimizer_config["params"])

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


def get_runner(config: dict, device: torch.device):
    if config.get("runner") is not None:
        if config["runner"] == "SAMRunner":
            return SAMRunner(device=device)
        else:
            raise NotImplementedError
    else:
        return SupervisedRunner(
            device=device,
            input_key=config["globals"]["input_key"],
            input_target_key=config["globals"]["input_target_key"])
