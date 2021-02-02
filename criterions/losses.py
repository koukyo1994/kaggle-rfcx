import torch
import torch.nn as nn
import torch.nn.functional as F

from torch.autograd import Function


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


class LSEP(Function):
    @staticmethod
    def forward(ctx, input, target):
        batch_size = target.size(0)

        positive_indices = target.gt(0).float()
        negative_indices = target.eq(0).float()
        loss = 0.0
        for i in range(input.size(0)):
            pos = torch.where(positive_indices[i])
            neg = torch.where(negative_indices[i])
            pos_inp = input[i][pos]
            neg_inp = input[i][neg]
            for p in pos_inp:
                loss += torch.exp(neg_inp - p).sum()

        loss = loss.log1p() / batch_size
        ctx.save_for_backward(input, target)
        ctx.loss = loss
        ctx.positive_indices = positive_indices
        ctx.negative_indices = negative_indices
        return loss

    @staticmethod
    def backward(ctx, grad_outputs):
        input, target = ctx.saved_variables
        loss = ctx.loss.detach()
        positive_indices = ctx.positive_indices
        negative_indices = ctx.negative_indices

        fac = -1.0 / loss
        grad_input = torch.zeros_like(input)

        for i in range(grad_input.size(0)):
            pos_ind = torch.where(positive_indices[i])[0]
            neg_ind = torch.where(negative_indices[i])[0]

            oh_pos = nn.functional.one_hot(pos_ind, input.size(1)).float()
            oh_neg = nn.functional.one_hot(neg_ind, input.size(1)).float()

            for j, phot in enumerate(oh_pos):
                for k, nhot in enumerate(oh_neg):
                    grad_input[i] += (phot - nhot) * torch.exp(-input[i].data * (phot - nhot))
        grad_input *= (grad_outputs * fac)
        return grad_input, None, None


class LSEPLoss(nn.Module):
    def __init__(self):
        super(LSEPLoss, self).__init__()

    def forward(self, input, target):
        return LSEP.apply(input, target)


class BCEFocalLoss(nn.Module):
    def __init__(self, gamma=2, weights=None):
        super().__init__()
        self.gamma = gamma
        if weights is None:
            self.weights = torch.tensor([1] * 24).float()
        else:
            self.weights = torch.tensor(weights).float()
        self.weights.requires_grad = False

        self.loss_fct = nn.BCEWithLogitsLoss(reduction="none")

    def forward(self, logit, target):
        if self.weights.device == torch.device("cpu"):
            self.weights = self.weights.to(logit.device)
        target = target.float()
        bce = self.loss_fct(logit, target)
        probas = torch.sigmoid(logit)
        loss = torch.where(target >= 0.5, (1.0 - probas) ** self.gamma * bce, probas ** self.gamma * bce)
        return loss.mean()


class FocalLoss(nn.Module):
    def __init__(self, gamma=2, weights=None):
        super().__init__()
        self.gamma = gamma
        if weights is None:
            self.weights = torch.tensor([1] * 24).float()
        else:
            self.weights = torch.tensor(weights).float()
        self.weights.requires_grad = False

    def forward(self, logit, target):
        if self.weights.device == torch.device("cpu"):
            self.weights = self.weights.to(logit.device)
        target = target.float()
        max_val = (-logit).clamp(min=0)
        loss = logit - logit * target + max_val + \
            ((-max_val).exp() + (-logit - max_val).exp()).log()

        invprobs = F.logsigmoid(-logit * (target * 2.0 - 1.0))
        loss = (invprobs * self.gamma).exp() * loss
        if len(loss.size()) == 2:
            loss = loss * self.weights
            loss = loss.sum(dim=1)
        return loss.mean()


class BCE2WayLoss(nn.Module):
    def __init__(self, output_key="clipwise_output", weights=[1, 1], class_weights=None):
        super().__init__()

        self.output_key = output_key
        if class_weights is not None:
            weight = torch.tensor(class_weights)
        else:
            weight = class_weights
        if "logit" in self.output_key:
            self.bce = nn.BCEWithLogitsLoss(weight=weight)
        else:
            self.bce = nn.BCELoss(weight=weight)

        self.weights = weights

    def forward(self, input, target):
        input_ = input[self.output_key]
        target = target["weak"].float()

        if self.bce.weight is not None and self.bce.weight.device == torch.device("cpu") and target.device != torch.device("cpu"):
            device = target.device
            self.bce.weight = self.bce.weight.to(device)

        if "logit" in self.output_key:
            framewise_output = input["framewise_logit"]
        else:
            framewise_output = input["framewise_output"]
        clipwise_output_with_max, _ = framewise_output.max(dim=1)

        loss = self.bce(input_, target)
        aux_loss = self.bce(clipwise_output_with_max, target)

        return self.weights[0] * loss + self.weights[1] * aux_loss


class BCE2WayStrongLoss(nn.Module):
    def __init__(self, output_key="framewise_output", weights=[1, 1]):
        super().__init__()

        self.output_key = output_key
        if "logit" in self.output_key:
            self.bce = nn.BCEWithLogitsLoss()
        else:
            self.bce = nn.BCELoss()

        self.weights = weights

    def forward(self, input, target):
        input_ = input[self.output_key]
        target_ = target["strong"].float()

        if "logit" in self.output_key:
            clipwise_output = input["logit"]
        else:
            clipwise_output = input["clipwise_output"]
        clipwise_target = target["weak"].float()

        loss = self.bce(input_, target_)
        aux_loss = self.bce(clipwise_output, clipwise_target)

        return self.weights[0] * loss + self.weights[1] * aux_loss


class BCEFocal2WayLoss(nn.Module):
    def __init__(self, weights=[1, 1], class_weights=None):
        super().__init__()

        self.focal = BCEFocalLoss(weights=class_weights)

        self.weights = weights

    def forward(self, input, target):
        input_ = input["logit"]
        target = target["weak"].float()

        framewise_output = input["framewise_logit"]
        clipwise_output_with_max, _ = framewise_output.max(dim=1)

        loss = self.focal(input_, target)
        aux_loss = self.focal(clipwise_output_with_max, target)

        return self.weights[0] * loss + self.weights[1] * aux_loss


class BCEFocal2WayStrongLoss(nn.Module):
    def __init__(self, weights=[1, 1], class_weights=None):
        super().__init__()

        self.focal = BCEFocalLoss(weights=class_weights)

        self.weights = weights

    def forward(self, input, target):
        input_ = input["framewise_logit"]
        target_ = target["strong"].float()

        clipwise_output = input["logit"]
        clipwise_target = target["weak"].float()

        loss = self.focal(input_, target_)
        aux_loss = self.focal(clipwise_output, clipwise_target)

        return self.weights[0] * loss + self.weights[1] * aux_loss


class Focal2WayLoss(nn.Module):
    def __init__(self, weights=[1, 1], class_weights=None):
        super().__init__()

        self.focal = FocalLoss(weights=class_weights)

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
    def __init__(self, weights=[1, 1], class_weights=None):
        super().__init__()

        self.focal = FocalLoss(weights=class_weights)

        self.weights = weights

    def forward(self, input, target):
        input_ = input["framewise_logit"]
        target_ = target["strong"].float()

        clipwise_output = input["logit"]
        clipwise_target = target["weak"].float()

        loss = self.focal(input_, target_)
        aux_loss = self.focal(clipwise_output, clipwise_target)

        return self.weights[0] * loss + self.weights[1] * aux_loss


class LogitLoss(nn.Module):
    def __init__(self):
        super().__init__()
        self.bce = nn.BCEWithLogitsLoss()

    def forward(self, input, target):
        input_ = input["logit"]
        target = target["weak"].float()
        loss = self.bce(input_, target)
        return loss


class LSEP2WayLoss(nn.Module):
    def __init__(self, output_key="clipwise_output", weights=[1, 1]):
        super().__init__()

        self.lsep = LSEPLoss()
        self.output_key = output_key
        self.weights = weights

    def forward(self, input, target):
        input_ = input[self.output_key]
        target = target["weak"].float()

        if "logit" in self.output_key:
            framewise_output = input["framewise_logit"]
        else:
            framewise_output = input["framewise_output"]
        clipwise_output_with_max, _ = framewise_output.max(dim=1)

        loss = self.lsep(input_, target)
        aux_loss = self.lsep(clipwise_output_with_max, target)

        return self.weights[0] * loss + self.weights[1] * aux_loss
