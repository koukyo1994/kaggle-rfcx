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
                   writer: SummaryWriter,
                   aggregate_by_recording=True):
    loss_meter = utils.AverageMeter()
    lwlrap_meter = utils.AverageMeter()

    model.eval()

    preds = []
    targs = []
    recording_ids = []
    indices = []
    progress_bar = tqdm(loader, desc="valid")
    for step, batch in enumerate(progress_bar):
        with torch.no_grad():
            recording_ids.extend(batch["recording_id"])
            if batch.get("index") is not None:
                with_index = True
                indices.extend(batch["index"].numpy())
            else:
                with_index = False
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
    if with_index:
        rec_df = pd.DataFrame({
            "recording_id": recording_ids,
            "index": indices
        })
    else:
        rec_df = pd.DataFrame({
            "recording_id": recording_ids,
            "index": indices
        })
    oof_pred_df = pd.concat([
        rec_df,
        oof_pred_df
    ], axis=1)

    oof_targ_df = pd.DataFrame(y_true, columns=[f"s{i}" for i in range(y_pred.shape[1])])
    oof_targ_df = pd.concat([
        rec_df,
        oof_targ_df
    ], axis=1)

    if aggregate_by_recording:
        oof_pred_df = oof_pred_df.groupby("recording_id").max().reset_index(drop=False)
        oof_targ_df = oof_targ_df.groupby("recording_id").max().reset_index(drop=False)

    columns = [f"s{i}" for i in range(24)]

    score_class, weight = clb.lwlrap(oof_targ_df[columns].values, oof_pred_df[columns].values)
    score = (score_class * weight).sum()

    writer.add_scalar(tag="loss/epoch", scalar_value=loss_meter.avg, global_step=epoch + 1)
    writer.add_scalar(tag="lwlrap/epoch", scalar_value=score, global_step=epoch + 1)
    return loss_meter.avg, score, oof_pred_df, oof_targ_df


def get_inference(model,
                  loader,
                  device: torch.device,
                  input_key: str,
                  input_target_key: str):
    recording_ids = []
    batch_predictions = []
    for batch in tqdm(loader, leave=True, desc="inference"):
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

    fold_prediction_df = fold_prediction_df.groupby("recording_id").max().reset_index(drop=False)
    return fold_prediction_df


def get_soft_inference(model,
                       loader,
                       device: torch.device,
                       input_key: str,
                       input_target_key: str):
    soft_prediction = {}
    for batch in tqdm(loader, leave=True, desc="soft inference"):
        recording_id = batch["recording_id"][0]
        input_ = batch[input_key].squeeze(0).to(device)
        with torch.no_grad():
            output = model(input_)
        framewise_output = output["framewise_output"].detach().cpu().numpy()
        clip_prediction = np.vstack(framewise_output)
        soft_prediction[recording_id] = clip_prediction
    return soft_prediction


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

    submission_file_dir = logdir / "submission"
    submission_file_dir.mkdir(exist_ok=True, parents=True)

    soft_oof_dir = logdir / "soft_oof"
    soft_oof_dir.mkdir(exist_ok=True, parents=True)

    soft_pred_dir = logdir / "soft_pred"
    soft_pred_dir.mkdir(exist_ok=True, parents=True)

    logger = utils.get_logger(logdir / "output.log")

    # environment
    utils.set_seed(global_params["seed"])
    device = training.get_device(global_params["device"])

    # data
    tp, fp, train_all, test_all, train_audio, test_audio = datasets.get_metadata(config)
    submission = pd.read_csv(config["data"]["sample_submission_path"])

    # validation
    splitter = training.get_split(config)

    ##################################################
    # Main Loop #
    ##################################################
    fold_predictions: dict = {"low": [], "high": []}
    oof_predictions: dict = {"low": [], "high": []}
    oof_targets: dict = {"low": [], "high": []}
    soft_oofs: dict = {}
    soft_preds: dict = {}
    for i, (trn_idx, val_idx) in enumerate(splitter.split(train_all)):
        if i not in global_params["folds"]:
            continue
        logger.info("=" * 20)
        logger.info(f"Fold {i}")
        logger.info("=" * 20)

        _logdir = logdir / f"fold{i}"
        _logdir.mkdir(exist_ok=True, parents=True)

        trn_df = train_all.loc[trn_idx, :].reset_index(drop=True)
        val_df = train_all.loc[val_idx, :].reset_index(drop=True)

        ##################################################
        # High Frequency Classes #
        ##################################################

        logger.info("Training for high frequency classes")

        _logdir_high = _logdir / "high"
        _logdir_high.mkdir(exist_ok=True, parents=True)

        checkpoints_dir = _logdir_high / "checkpoints"
        checkpoints_dir.mkdir(exist_ok=True, parents=True)

        train_writer = SummaryWriter(log_dir=_logdir_high / "train_log")
        valid_writer = SummaryWriter(log_dir=_logdir_high / "valid_log")

        config["dataset"]["train"]["params"]["frequency_range"] = "high"
        config["dataset"]["valid"]["params"]["frequency_range"] = "high"
        config["dataset"]["test"]["params"]["frequency_range"] = "high"

        loaders = {
            phase: datasets.get_train_loader(df_, tp, fp, train_audio, config, phase)
            for df_, phase in zip([trn_df, val_df], ["train", "valid"])
        }
        test_loader = datasets.get_test_loader(test_all, test_audio, config)

        soft_inference_config = {
            "loader": {
                "test": {
                    "batch_size": 1,
                    "shuffle": False,
                    "num_workers": config["loader"]["test"]["num_workers"]
                }
            },
            "dataset": {
                "test": {
                    "name": "LimitedFrequencySampleWiseSpectrogramTestDataset",
                    "params": config["dataset"]["test"]["params"]
                }
            },
            "transforms": None
        }

        val_soft_loader = datasets.get_test_loader(val_df, train_audio, soft_inference_config)
        test_soft_loader = datasets.get_test_loader(test_all, test_audio, soft_inference_config)

        model = models.get_model(config).to(device)
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

            valid_loss, valid_score, _, _ = eval_one_epoch(
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

        model = models.prepare_for_inference(model, checkpoints_dir / "best.pth").to(device)
        aggregate_by_recording = config["dataset"]["valid"]["name"] == "LimitedFrequencySequentialValidationDataset"
        _, _, oof_pred_df, oof_targ_df = eval_one_epoch(
            model,
            loaders["valid"],
            criterion,
            device,
            input_key=global_params["input_key"],
            input_target_key=global_params["input_target_key"],
            epoch=epoch + 1,
            writer=valid_writer,
            aggregate_by_recording=aggregate_by_recording)

        oof_predictions["high"].append(oof_pred_df)
        oof_targets["high"].append(oof_targ_df)

        fold_prediction = get_inference(
            model, test_loader, device,
            input_key=global_params["input_key"],
            input_target_key=global_params["input_target_key"])

        fold_predictions["high"].append(fold_prediction)

        soft_oof = get_soft_inference(
            model,
            loader=val_soft_loader,
            device=device,
            input_key=global_params["input_key"],
            input_target_key=global_params["input_target_key"])
        high_freq_keys = datasets.RANGE_SPECIES_MAP["high"]
        for key in soft_oof.keys():
            soft_oofs[key] = {}
            soft_oof_pred = soft_oof[key]
            for high_freq_key in high_freq_keys:
                soft_oofs[key][high_freq_key] = soft_oof_pred[:, high_freq_key]

        soft_pred = get_soft_inference(
            model,
            loader=test_soft_loader,
            device=device,
            input_key=global_params["input_key"],
            input_target_key=global_params["input_target_key"])
        soft_preds[f"fold{i}"] = {}
        for key in soft_pred.keys():
            soft_preds[f"fold{i}"][key] = {}
            soft_test_pred = soft_pred[key]
            for high_freq_key in high_freq_keys:
                soft_preds[f"fold{i}"][key][high_freq_key] = soft_test_pred[:, high_freq_key]

        train_writer.close()
        valid_writer.close()

        ##################################################
        # Low Frequency Classes #
        ##################################################

        logger.info("Training for low frequency classes")

        _logdir_low = _logdir / "low"
        _logdir_low.mkdir(exist_ok=True, parents=True)

        checkpoints_dir = _logdir_low / "checkpoints"
        checkpoints_dir.mkdir(exist_ok=True, parents=True)

        train_writer = SummaryWriter(log_dir=_logdir_low / "train_log")
        valid_writer = SummaryWriter(log_dir=_logdir_low / "valid_log")

        config["dataset"]["train"]["params"]["frequency_range"] = "low"
        config["dataset"]["valid"]["params"]["frequency_range"] = "low"
        config["dataset"]["test"]["params"]["frequency_range"] = "low"

        loaders = {
            phase: datasets.get_train_loader(df_, tp, fp, train_audio, config, phase)
            for df_, phase in zip([trn_df, val_df], ["train", "valid"])
        }
        test_loader = datasets.get_test_loader(test_all, test_audio, config)

        soft_inference_config = {
            "loader": {
                "test": {
                    "batch_size": 1,
                    "shuffle": False,
                    "num_workers": config["loader"]["test"]["num_workers"]
                }
            },
            "dataset": {
                "test": {
                    "name": "LimitedFrequencySampleWiseSpectrogramTestDataset",
                    "params": config["dataset"]["test"]["params"]
                }
            },
            "transforms": None
        }

        val_soft_loader = datasets.get_test_loader(val_df, train_audio, soft_inference_config)
        test_soft_loader = datasets.get_test_loader(test_all, test_audio, soft_inference_config)

        model = models.get_model(config).to(device)
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

            valid_loss, valid_score, _, _ = eval_one_epoch(
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

        model = models.prepare_for_inference(model, checkpoints_dir / "best.pth").to(device)
        aggregate_by_recording = config["dataset"]["valid"]["name"] == "LimitedFrequencySequentialValidationDataset"
        _, _, oof_pred_df, oof_targ_df = eval_one_epoch(
            model,
            loaders["valid"],
            criterion,
            device,
            input_key=global_params["input_key"],
            input_target_key=global_params["input_target_key"],
            epoch=epoch + 1,
            writer=valid_writer,
            aggregate_by_recording=aggregate_by_recording)

        oof_predictions["low"].append(oof_pred_df)
        oof_targets["low"].append(oof_targ_df)

        fold_prediction = get_inference(
            model, test_loader, device,
            input_key=global_params["input_key"],
            input_target_key=global_params["input_target_key"])

        fold_predictions["low"].append(fold_prediction)

        soft_oof = get_soft_inference(
            model,
            loader=val_soft_loader,
            device=device,
            input_key=global_params["input_key"],
            input_target_key=global_params["input_target_key"])
        low_freq_keys = datasets.RANGE_SPECIES_MAP["low"]
        for key in soft_oof.keys():
            soft_oof_pred = soft_oof[key]
            for low_freq_key in low_freq_keys:
                if soft_oofs[key].get(low_freq_key) is not None:
                    soft_oofs[key][low_freq_key] = (soft_oofs[key][low_freq_key] + soft_oof_pred[:, low_freq_key]) / 2
                else:
                    soft_oofs[key][low_freq_key] = soft_oof_pred[:, low_freq_key]

        soft_pred = get_soft_inference(
            model,
            loader=test_soft_loader,
            device=device,
            input_key=global_params["input_key"],
            input_target_key=global_params["input_target_key"])
        for key in soft_pred.keys():
            soft_test_pred = soft_pred[key]
            for low_freq_key in low_freq_keys:
                if soft_preds[f"fold{i}"][key].get(low_freq_key) is not None:
                    soft_preds[f"fold{i}"][key][low_freq_key] = (
                        soft_preds[f"fold{i}"][key][low_freq_key] +
                        soft_test_pred[:, low_freq_key]
                    ) / 2
                else:
                    soft_preds[f"fold{i}"][key][low_freq_key] = soft_test_pred[:, low_freq_key]

        train_writer.close()
        valid_writer.close()

    for key in soft_oofs:
        np.savez_compressed(soft_oof_dir / key, soft_oofs[key])

    for fold_key in soft_preds:
        (soft_pred_dir / fold_key).mkdir(exist_ok=True, parents=True)
        for key in soft_preds[fold_key]:
            np.savez_compressed(soft_pred_dir / fold_key / key, soft_preds[fold_key][key])

    oof_df_high = pd.concat(oof_predictions["high"], axis=0).reset_index(drop=True)
    oof_target_high = pd.concat(oof_targets["high"], axis=0).reset_index(drop=True)

    oof_df_low = pd.concat(oof_predictions["low"], axis=0).reset_index(drop=True)
    oof_target_low = pd.concat(oof_targets["low"], axis=0).reset_index(drop=True)

    if "index" in oof_df_high.columns:
        with_index = True
        rec_df = oof_df_high[["recording_id", "index"]]
    else:
        with_index = False
        rec_df = oof_df_high[["recording_id"]]

    oof_df = pd.concat([
        rec_df,
        pd.DataFrame(np.zeros((len(oof_df_high), 24)), columns=[f"s{i}" for i in range(24)])
    ], axis=1)

    oof_targets_df = pd.concat([
        rec_df,
        pd.DataFrame(np.zeros((len(oof_df_high), 24)), columns=[f"s{i}" for i in range(24)])
    ], axis=1)

    folds_prediction_high = pd.concat(fold_predictions["high"], axis=0).reset_index(drop=True)
    folds_prediction_low = pd.concat(fold_predictions["low"], axis=0).reset_index(drop=True)

    folds_prediction_df = pd.concat([
        folds_prediction_high[["recording_id"]],
        pd.DataFrame(np.zeros((len(folds_prediction_high), 24)), columns=[f"s{i}" for i in range(24)])
    ], axis=1)

    for i in range(24):
        species_id = f"s{i}"
        if datasets.SPECIES_RANGE_MAP[i] == ["high"]:
            oof_df[species_id] = oof_df_high[species_id]
            folds_prediction_df[species_id] = folds_prediction_high[species_id]
            oof_targets_df[species_id] = oof_target_high[species_id]
        elif datasets.SPECIES_RANGE_MAP[i] == ["low"]:
            oof_df[species_id] = oof_df_low[species_id]
            folds_prediction_df[species_id] = folds_prediction_low[species_id]
            oof_targets_df[species_id] = oof_target_low[species_id]
        else:
            oof_df[species_id] = 0.5 * oof_df_high[species_id] + 0.5 * oof_df_low[species_id]
            folds_prediction_df[species_id] = 0.5 * folds_prediction_high[species_id] + 0.5 * folds_prediction_low[species_id]
            oof_targets_df[species_id] = 0.5 * oof_target_high[species_id] + 0.5 * oof_target_low[species_id]

    folds_prediction_df = folds_prediction_df.groupby("recording_id").mean().reset_index(drop=False)

    oof_df_high.to_csv(submission_file_dir / "oof_high.csv", index=False)
    oof_target_high.to_csv(submission_file_dir / "oof_target_high.csv", index=False)
    oof_df_low.to_csv(submission_file_dir / "oof_low.csv", index=False)
    oof_target_low.to_csv(submission_file_dir / "oof_target_low.csv", index=False)
    oof_df.to_csv(submission_file_dir / "oof.csv", index=False)
    oof_targets_df.to_csv(submission_file_dir / "oof_target.csv", index=False)

    assert len(folds_prediction_df) == len(submission), \
        "prediction length does not match sample submission length"
    assert folds_prediction_df.shape[1] == submission.shape[1], \
        "number of classes in prediction does not match that of sample submission"
    assert len(set(folds_prediction_df["recording_id"]) - set(submission["recording_id"])) == 0, \
        "recording_id in prediction has unknown value"
    assert len(set(submission["recording_id"]) - set(folds_prediction_df["recording_id"])) == 0, \
        "prediction doesn't have enough recording_id"

    folds_prediction_df.to_csv(submission_file_dir / "submission.csv", index=False)

    summary = {}

    columns = [f"s{i}" for i in range(24)]
    score_class, weight = clb.lwlrap(oof_target_high[columns].values, oof_df_high[columns].values)
    score_high = (score_class * weight).sum()
    logger.info(f"Valid LWLRAP(high): {score_high:.5f}")

    summary["high"] = {
        "score": score_high,
        "score_class": score_class,
        "weight": weight
    }

    score_class, weight = clb.lwlrap(oof_target_low[columns].values, oof_df_low[columns].values)
    score_low = (score_class * weight).sum()
    logger.info(f"Valid LWLRAP(low): {score_low:.5f}")

    summary["low"] = {
        "score": score_low,
        "score_class": score_class,
        "weight": weight
    }

    score_class, weight = clb.lwlrap(oof_targets_df[columns].values, oof_df[columns].values)
    score = (score_class * weight).sum()
    logger.info(f"Valid LWLRAP(all): {score:.5f}")

    summary["all"] = {
        "score": score,
        "score_class": score_class,
        "weight": weight
    }

    utils.save_json(summary, submission_file_dir / "results.json")
