import subprocess as sp
import torch

from pathlib import Path

from .effcientnet import EfficientNetSED, TimmEfficientNetSED
from .layers import AttBlock, AttBlockV2
from .panns import PANNsCNN14Att
from .resnest import ResNestSED
from .utils import init_layer


def get_model(config: dict):
    model_config = config["model"]
    model_name = model_config["name"]
    model_params = model_config["params"]

    if model_name == "PANNsCNN14Att":
        if model_params["pretrained"]:
            model = PANNsCNN14Att(  # type: ignore
                sample_rate=32000,
                window_size=1024,
                hop_size=320,
                mel_bins=64,
                fmin=50,
                fmax=14000,
                classes_num=527)
            checkpoint_path = Path("bin/Cnn14_DecisionLevelAtt_mAP0.425.pth")
            if not checkpoint_path.exists():
                cmd = [
                    "kaggle", "datasets", "download",
                    "-d", "hidehisaarai1213/pannscnn14-decisionlevelatt-weight",
                    "-p", "bin/", "--unzip"]
                sp.run(cmd, capture_output=False)
            checkpoint = torch.load(checkpoint_path)
            model.load_state_dict(checkpoint["model"])

            model.att_block = AttBlock(
                2048, model_params["n_classes"], activation="sigmoid")
            model.att_block.init_weights()
            init_layer(model.fc1)
        else:
            model = PANNsCNN14Att(**model_params)  # type: ignore

        weights_path = config["globals"].get("weights")
        if weights_path is not None:
            if Path(weights_path).exists():
                weights = torch.load(weights_path)["model_state_dict"]
                # to fit for birdcall competition
                n_classes = weights["att_block.att.weight"].size(0)
                model.att_block = AttBlock(
                    2048, n_classes, activation="sigmoid")
                model.load_state_dict(weights)
                model.att_block = AttBlock(
                    2048, model_params["n_classes"], activation="sigmoid")
                model.att_block.init_weights()
        return model
    elif model_name == "ResNestSED":
        model = ResNestSED(**model_params)  # type: ignore

        weights_path = config["globals"].get("weights")
        if weights_path is not None:
            if Path(weights_path).exists():
                weights = torch.load(weights_path)["model_state_dict"]
                # for loading ema weight
                model_state_dict = {}
                for key in weights:
                    if key == "n_averaged":
                        continue
                    new_key = key.replace("module.", "")
                    model_state_dict[new_key] = weights[key]
                # to fit for birdcall competition
                n_classes = model_state_dict["att_block.att.weight"].size(0)
                model.att_block = AttBlockV2(  # type: ignore
                    2048, n_classes, activation="sigmoid")
                model.load_state_dict(model_state_dict)
                model.att_block = AttBlockV2(  # type: ignore
                    2048, model_params["num_classes"], activation="sigmoid")
                model.att_block.init_weights()
        return model
    elif model_name == "EfficientNetSED":
        model = EfficientNetSED(**model_params)  # type: ignore

        weights_path = config["globals"].get("weights")
        if weights_path is not None:
            if Path(weights_path).exists():
                weights = torch.load(weights_path)["model_state_dict"]
                # for loading ema weight
                model_state_dict = {}
                for key in weights:
                    if key == "n_averaged":
                        continue
                    new_key = key.replace("module.", "")
                    model_state_dict[new_key] = weights[key]
                # to fit for birdcall competition
                n_classes = model_state_dict["att_block.att.weight"].size(0)
                model.att_block = AttBlockV2(  # type: ignore
                    2048, n_classes, activation="sigmoid")
                model.load_state_dict(model_state_dict)
                model.att_block = AttBlockV2(  # type: ignore
                    2048, model_params["num_classes"], activation="sigmoid")
                model.att_block.init_weights()
        return model
    elif model_name == "TimmEfficientNetSED":
        model = TimmEfficientNetSED(**model_params)  # type: ignore

        weights_path = config["globals"].get("weights")
        if weights_path is not None:
            if Path(weights_path).exists():
                weights = torch.load(weights_path)["model_state_dict"]
                # for loading ema weight
                model_state_dict = {}
                for key in weights:
                    if key == "n_averaged":
                        continue
                    new_key = key.replace("module.", "")
                    model_state_dict[new_key] = weights[key]
                # to fit for birdcall competition
                n_classes = model_state_dict["att_block.att.weight"].size(0)
                model.att_block = AttBlockV2(  # type: ignore
                    2048, n_classes, activation="sigmoid")
                model.load_state_dict(model_state_dict)
                model.att_block = AttBlockV2(  # type: ignore
                    2048, model_params["num_classes"], activation="sigmoid")
                model.att_block.init_weights()
        return model
    else:
        raise NotImplementedError


def prepare_for_inference(model, checkpoint_path: Path):
    if not torch.cuda.is_available():
        checkpoint = torch.load(checkpoint_path, map_location="cpu")
    else:
        checkpoint = torch.load(checkpoint_path)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()
    return model
