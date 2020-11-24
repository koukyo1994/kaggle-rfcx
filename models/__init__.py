import subprocess as sp
import torch

from pathlib import Path

from .layers import AttBlock
from .panns import PANNsCNN14Att
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
                weights = torch.load(weights_path)
                model.load_state_dict(weights["model_state_dict"])
        return model
    else:
        raise NotImplementedError
