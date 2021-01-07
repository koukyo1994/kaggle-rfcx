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

from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm


def train_one_epoch(model,
                    loader,
                    optimizer,
                    scheduler,
                    criterion,
                    device: torch.device,
                    input_key: str,
                    input_target_key: str,
                    epoch: int,
                    writer: SummaryWriter):
    loss_meter = utils.AverageMeter()
    lwlrap_meter = utils.AverageMeter()

    model.train()

    preds = []
    targs = []

    progress_bar = tqdm(loader, desc="train")
    for step, batch in enumerate(progress_bar):
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

    y_pred = np.concatenate(preds, axis=0)
    y_true = np.concatenate(targs, axis=0)

    score_class, weight = clb.lwlrap(y_true, y_pred)
    score = (score_class * weight).sum()

    writer.add_scalar(tag="loss/epoch", scalar_value=loss_meter.avg, global_step=epoch + 1)
    writer.add_scalar(tag="lwlrap/epoch", scalar_value=score, global_step=epoch + 1)
    return loss_meter.avg, score


def eval_one_epoch(model,
                   loader,
                   criterion,
                   device: torch.device,
                   input_key: str,
                   input_target_key: str,
                   epoch: int,
                   writer: SummaryWriter):
    loss_meter = utils.AverageMeter()
    lwlrap_meter = utils.AverageMeter()

    model.eval()

    preds = []
    targs = []
    recording_ids = []
    progress_bar = tqdm(loader, desc="valid")
    for step, batch in enumerate(progress_bar):
        with torch.no_grad():
            recording_ids.extend(batch["recording_id"])
            x = batch[input_key].to(device)
            y = batch[input_target_key]

            for key in y:
                y[key] = y[key].to(device)

            output = model(x)
            loss = criterion(output, y).detach()

        loss_meter.update(loss.item(), n=len(loader))

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

    y_pred = np.concatenate(preds, axis=0)
    y_true = np.concatenate(targs, axis=0)

    oof_pred_df = pd.DataFrame(y_pred, columns=[f"s{i}" for i in range(y_pred.shape[1])])
    oof_pred_df = pd.concat([
        pd.DataFrame({"recording_id": recording_ids}),
        oof_pred_df
    ], axis=1)

    oof_targ_df = pd.DataFrame(y_true, columns=[f"s{i}" for i in range(y_pred.shape[1])])
    oof_targ_df = pd.concat([
        pd.DataFrame({"recording_id": recording_ids}),
        oof_targ_df
    ], axis=1)

    oof_pred_df = oof_pred_df.groupby("recording_id").max().reset_index(drop=True)
    oof_targ_df = oof_targ_df.groupby("recording_id").max().reset_index(drop=True)

    score_class, weight = clb.lwlrap(oof_targ_df.values, oof_pred_df.values)
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
    tp, fp, train_all, _, train_audio, _ = datasets.get_metadata(config)
    # validation
    splitter = training.get_split(config)

    ##################################################
    # Main Loop #
    ##################################################
    for i, (trn_idx, val_idx) in enumerate(splitter.split(train_all)):
        if i not in global_params["folds"]:
            continue
        logger.info("=" * 20)
        logger.info(f"Fold {i}")
        logger.info("=" * 20)

        _logdir = logdir / f"fold{i}"
        _logdir.mkdir(exist_ok=True, parents=True)

        checkpoints_dir = _logdir / "checkpoints"
        checkpoints_dir.mkdir(exist_ok=True, parents=True)

        train_writer = SummaryWriter(log_dir=_logdir / "train_log")
        valid_writer = SummaryWriter(log_dir=_logdir / "valid_log")

        trn_df = train_all.loc[trn_idx, :].reset_index(drop=True)
        val_df = train_all.loc[val_idx, :].reset_index(drop=True)

        loaders = {
            phase: datasets.get_train_loader(df_, tp, fp, train_audio, config, phase)
            for df_, phase in zip([trn_df, val_df], ["train", "valid"])
        }
        model = models.get_model(config, fold=i).to(device)
        criterion = criterions.get_criterion(config)
        optimizer = training.get_optimizer(model, config)
        scheduler = training.get_scheduler(optimizer, config)

        best_score = 0.0
        _metrics = {}
        for epoch in range(global_params["num_epochs"]):
            logger.info(f"Epoch: [{epoch+1}/{global_params['num_epochs']}]")
            train_loss, train_score = train_one_epoch(
                model,
                loaders["train"],
                optimizer,
                scheduler,
                criterion,
                device,
                epoch=epoch,
                input_key=global_params["input_key"],
                input_target_key=global_params["input_target_key"],
                writer=train_writer)

            valid_loss, valid_score = eval_one_epoch(
                model,
                loaders["valid"],
                criterion,
                device,
                input_key=global_params["input_key"],
                input_target_key=global_params["input_target_key"],
                epoch=epoch,
                writer=valid_writer)

            best_score, updated = utils.save_best_model(
                model, checkpoints_dir, valid_score, prev_metric=best_score)

            if updated:
                _metrics["best"] = {"lwlrap": best_score, "loss": valid_loss, "epoch": epoch + 1}
            _metrics["last"] = {"lwlrap": valid_score, "loss": valid_loss, "epoch": epoch + 1}
            _metrics[f"epoch_{epoch + 1}"] = {"lwlrap": valid_score, "loss": valid_loss}

            utils.save_json(_metrics, checkpoints_dir / "_metrics.json")

            logger.info(
                f"{epoch + 1}/{global_params['num_epochs']} * Epoch {epoch + 1} "
                f"(train): lwlrap={train_score:.4f} | loss={train_loss:.4f}")
            logger.info(
                f"{epoch + 1}/{global_params['num_epochs']} * Epoch {epoch + 1} "
                f"(valid): lwlrap={valid_score:.4f} | loss={valid_loss:.4f}")
        logger.info(
            f"Best epoch: {_metrics['best']['epoch']} lwlrap: {_metrics['best']['lwlrap']} loss: {_metrics['best']['loss']}")

        train_writer.close()
        valid_writer.close()
