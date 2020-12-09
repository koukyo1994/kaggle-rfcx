import warnings

import numpy as np
import pandas as pd
import torch

import callbacks as clb
import criterions
import datasets
import models
import training
import utils

from pathlib import Path

from torch.optim.swa_utils import AveragedModel
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm


def update_bn(loader, model, device=None, input_key=""):
    r"""Updates BatchNorm running_mean, running_var buffers in the model.
    It performs one pass over data in `loader` to estimate the activation
    statistics for BatchNorm layers in the model.
    Arguments:
        loader (torch.utils.data.DataLoader): dataset loader to compute the
            activation statistics on. Each data batch should be either a
            tensor, or a list/tuple whose first element is a tensor
            containing data.
        model (torch.nn.Module): model for which we seek to update BatchNorm
            statistics.
        device (torch.device, optional): If set, data will be transferred to
            :attr:`device` before being passed into :attr:`model`.
    Example:
        >>> loader, model = ...
        >>> torch.optim.swa_utils.update_bn(loader, model)
    .. note::
        The `update_bn` utility assumes that each data batch in :attr:`loader`
        is either a tensor or a list or tuple of tensors; in the latter case it
        is assumed that :meth:`model.forward()` should be called on the first
        element of the list or tuple corresponding to the data batch.
    """
    momenta = {}
    for module in model.modules():
        if isinstance(module, torch.nn.modules.batchnorm._BatchNorm):
            module.running_mean = torch.zeros_like(module.running_mean)
            module.running_var = torch.ones_like(module.running_var)
            momenta[module] = module.momentum

    if not momenta:
        return

    was_training = model.training
    model.train()
    for module in momenta.keys():
        module.momentum = None
        module.num_batches_tracked *= 0

    for input in loader:
        if isinstance(input, (list, tuple)):
            input = input[0]
        if isinstance(input, dict):
            input = input[input_key]
        if device is not None:
            input = input.to(device)

        model(input)

    for bn_module in momenta.keys():
        bn_module.momentum = momenta[bn_module]
    model.train(was_training)


def train_one_epoch(model,
                    ema_model,
                    loader,
                    optimizer,
                    scheduler,
                    criterion,
                    device: torch.device,
                    input_key: str,
                    input_target_key: str,
                    epoch: int,
                    writer: SummaryWriter,
                    ema_update_interval=10):
    loss_meter = utils.AverageMeter()
    lwlrap_meter = utils.AverageMeter()

    model.train()

    preds = []
    targs = []

    count = ema_update_interval

    progress_bar = tqdm(loader, desc="train")
    for step, batch in enumerate(progress_bar):
        count -= 1

        x = batch[input_key].to(device)
        y = batch[input_target_key]
        for key in y:
            y[key] = y[key].to(device)

        output = model(x)
        loss = criterion(output, y)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        loss_meter.update(loss.item(), n=len(loader))

        if count == 0:
            ema_model.update_parameters(model)
            count = ema_update_interval

        clipwise_output = output["clipwise_output"].detach().cpu().numpy()
        target = y["weak"].detach().cpu().numpy()

        preds.append(clipwise_output)
        targs.append(target)

        score_class, weight = clb.lwlrap(target, clipwise_output)
        score = (score_class * weight).sum()
        lwlrap_meter.update(score, n=1)

        progress_bar.set_description(
            f"Epoch: {epoch + 1} "
            f"Step: [{step + 1}/{len(loader)}] "
            f"loss: {loss_meter.val:.4f} loss(avg) {loss_meter.avg:.4f} "
            f"lwlrap: {lwlrap_meter.val:.4f} lwlrap(avg) {lwlrap_meter.avg:.4f}")

        global_step = epoch * len(loader) + step + 1
        writer.add_scalar(tag="loss/batch", scalar_value=loss_meter.val, global_step=global_step)
        writer.add_scalar(tag="lwlrap/batch", scalar_value=lwlrap_meter.val, global_step=global_step)

    scheduler.step()

    update_bn(loader, ema_model, device=device, input_key=input_key)

    y_pred = np.concatenate(preds, axis=0)
    y_true = np.concatenate(targs, axis=0)

    score_class, weight = clb.lwlrap(y_true, y_pred)
    score = (score_class * weight).sum()

    writer.add_scalar(tag="loss/epoch", scalar_value=loss_meter.avg, global_step=epoch + 1)
    writer.add_scalar(tag="lwlrap/epoch", scalar_value=score, global_step=epoch + 1)
    return loss_meter.avg, score


if __name__ == "__main__":
    warnings.filterwarnings("ignore")

    ##################################################
    #  Basic configuration #
    ##################################################
    args = utils.get_parser().parse_args()
    config = utils.load_config(args.config)

    global_params = config["globals"]

    # logging
    config_name = args.config.split("/")[-1].replace(".yml", "")
    logdir = Path(f"out/{config_name}")
    logdir.mkdir(exist_ok=True, parents=True)

    logger = utils.get_logger(logdir / "output.log")

    # environment
    utils.set_seed(global_params["seed"])
    device = training.get_device(global_params["device"])

    # data
    tp, fp, train_all, test_all, train_audio, test_audio = datasets.get_metadata(config)
    submission = pd.read_csv(config["data"]["sample_submission_path"])

    ##################################################
    # Main Loop #
    ##################################################
    logger.info("=" * 20)
    logger.info("No Fold Training")
    logger.info("=" * 20)

    checkpoints_dir = logdir / "checkpoints"
    checkpoints_dir.mkdir(exist_ok=True, parents=True)

    train_writer = SummaryWriter(log_dir=logdir / "train_log")

    loader = datasets.get_train_loader(train_all, tp, fp, train_audio, config, phase="train")

    model = models.get_model(config).to(device)
    criterion = criterions.get_criterion(config)
    optimizer = training.get_optimizer(model, config)
    scheduler = training.get_scheduler(optimizer, config)

    ema_model = AveragedModel(
        model,
        device=device,
        avg_fn=lambda averaged_model_parameter, model_parameter, num_averaged:
            0.1 * averaged_model_parameter + 0.9 * model_parameter)

    _metrics = {}
    for epoch in range(global_params["num_epochs"]):
        logger.info(f"Epoch: [{epoch+1}/{global_params['num_epochs']}]")
        train_loss, train_score = train_one_epoch(
            model,
            ema_model,
            loader,
            optimizer,
            scheduler,
            criterion,
            device,
            epoch=epoch,
            input_key=global_params["input_key"],
            input_target_key=global_params["input_target_key"],
            writer=train_writer)

        utils.save_model(ema_model, checkpoints_dir, "ema.pth")

        _metrics["last"] = {"lwlrap": train_score, "loss": train_loss, "epoch": epoch + 1}
        _metrics[f"epoch_{epoch + 1}"] = {"lwlrap": train_score, "loss": train_loss}

        utils.save_json(_metrics, checkpoints_dir / "_metrics.json")

        logger.info(
            f"{epoch + 1}/{global_params['num_epochs']} * Epoch {epoch + 1} "
            f"(train): lwlrap={train_score:.4f} | loss={train_loss:.4f}")

    train_writer.close()

    submission_file_dir = logdir / "submission"
    submission_file_dir.mkdir(exist_ok=True, parents=True)

    #################################################
    # Prediction #
    ##################################################
    logger.info("*" * 20)
    logger.info("Prediction")
    logger.info("*" * 20)

    loader = datasets.get_test_loader(test_all, test_audio, config)

    recording_ids = []
    batch_predictions = []
    for batch in tqdm(loader):
        recording_ids.extend(batch["recording_id"])
        input_ = batch[global_params["input_key"]].to(device)
        with torch.no_grad():
            output = model(input_)
        batch_predictions.append(
            output["clipwise_output"].detach().cpu().numpy())
    fold_prediction = np.concatenate(batch_predictions, axis=0)

    fold_prediction_df = pd.DataFrame(
        fold_prediction, columns=[f"s{i}" for i in range(fold_prediction.shape[1])])
    fold_prediction_df = pd.concat([
        pd.DataFrame({"recording_id": recording_ids}),
        fold_prediction_df
    ], axis=1)

    fold_prediction_df = fold_prediction_df.groupby(
        "recording_id").max().reset_index(drop=False)

    assert len(fold_prediction_df) == len(submission), \
        "prediction length does not match sample submission length"
    assert fold_prediction_df.shape[1] == submission.shape[1], \
        "number of classes in prediction does not match that of sample submission"
    assert len(set(fold_prediction_df["recording_id"]) - set(submission["recording_id"])) == 0, \
        "recording_id in prediction has unknown value"
    assert len(set(submission["recording_id"]) - set(fold_prediction_df["recording_id"])) == 0, \
        "prediction doesn't have enough recording_id"

    fold_prediction_df.to_csv(submission_file_dir / "weak.csv", index=False)
