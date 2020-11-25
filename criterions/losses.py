import torch.nn as nn
import torch.nn.functional as F


def get_criterion(config: dict):
    loss_config = config["loss"]
    loss_name = loss_config["name"]
    loss_params = {} if loss_config.get("params") is None else loss_config.get(
        "params")

    if hasattr(nn, loss_name):
        criterion = nn.__getattribute__(loss_name)(**loss_params)
    else:
        criterion_cls = globals().get(loss_name)
        if criterion_cls is not None:
            criterion = criterion_cls(**loss_params)
        else:
            raise NotImplementedError

    return criterion


class FocalLoss(nn.Module):
    def __init__(self, gamma=2):
        super().__init__()
        self.gamma = gamma

    def forward(self, logit, target):
        target = target.float()
        max_val = (-logit).clamp(min=0)
        loss = logit - logit * target + max_val + \
            ((-max_val).exp() + (-logit - max_val).exp()).log()

        invprobs = F.logsigmoid(-logit * (target * 2.0 - 1.0))
        loss = (invprobs * self.gamma).exp() * loss
        if len(loss.size()) == 2:
            loss = loss.sum(dim=1)
        return loss.mean()


class BCE2WayLoss(nn.Module):
    def __init__(self, output_key="clipwise_output", weights=[1, 1]):
        super().__init__()

        self.output_key = output_key
        self.bce = nn.BCELoss()

        self.weights = weights

    def forward(self, input, target):
        input_ = input[self.output_key]
        target = target["weak"].float()

        framewise_output = input["framewise_output"]
        clipwise_output_with_max, _ = framewise_output.max(dim=1)

        loss = self.bce(input_, target)
        aux_loss = self.bce(clipwise_output_with_max, target)

        return self.weights[0] * loss + self.weights[1] * aux_loss


class BCE2WayStrongLoss(nn.Module):
    def __init__(self, output_key="framewise_output", weights=[1, 1]):
        super().__init__()

        self.output_key = output_key
        self.bce = nn.BCELoss()

        self.weights = weights

    def forward(self, input, target):
        input_ = input[self.output_key]
        target = target["strong"].float()

        clipwise_output = input["clipwise_output"]
        clipwise_target, _ = target.max(dim=1)

        loss = self.bce(input_, target)
        aux_loss = self.bce(clipwise_output, clipwise_target)

        return self.weights[0] * loss + self.weights[1] * aux_loss


class Focal2WayLoss(nn.Module):
    def __init__(self, weights=[1, 1]):
        super().__init__()

        self.focal = FocalLoss()

        self.weights = weights

    def forward(self, input, target):
        input_ = input["logit"]
        target = target["weak"].float()

        framewise_output = input["framewise_logit"]
        clipwise_output_with_max, _ = framewise_output.max(dim=1)

        loss = self.focal(input_, target)
        aux_loss = self.focal(clipwise_output_with_max, target)

        return self.weights[0] * loss + self.weights[1] * aux_loss


class Focal2WayStrongLoss(nn.Module):
    def __init__(self, weights=[1, 1]):
        super().__init__()

        self.focal = FocalLoss()

        self.weights = weights

    def forward(self, input, target):
        input_ = input["framewise_logit"]
        target = target["strong"].float()

        clipwise_output = input["logit"]
        clipwise_target, _ = target.max(dim=1)

        loss = self.focal(input_, target)
        aux_loss = self.focal(clipwise_output, clipwise_target)

        return self.weights[0] * loss + self.weights[1] * aux_loss
