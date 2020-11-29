import sys

import numpy as np
import pandas as pd
import torch

import datasets
import models
import training
import utils

from pathlib import Path

from tqdm import tqdm


if __name__ == "__main__":
    ##################################################
    #  Basic configuration #
    ##################################################
    args = utils.get_parser().parse_args()
    config = utils.load_config(args.config)

    global_params = config["globals"]

    # logging
    config_name = args.config.split("/")[-1].replace(".yml", "")
    expdir = Path(f"out/{config_name}")
    if not expdir.exists():
        print(f"You need to train {config_name} first!")
        sys.exit(1)
    submission_file_dir = expdir / "submission"
    submission_file_dir.mkdir(parents=True, exist_ok=True)

    logger = utils.get_logger(expdir / "inference.log")

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
    fold_predictions = []
    oof_predictions = []
    for i, (trn_idx, val_idx) in enumerate(splitter.split(train_all)):
        if i not in global_params["folds"]:
            continue
        logger.info("=" * 20)
        logger.info(f"Fold {i}")
        logger.info("=" * 20)

        val_df = train_all.iloc[val_idx, :].reset_index(drop=True)
        val_loader = datasets.get_train_loader(
            val_df, tp, fp, train_audio, config, phase="valid")

        loader = datasets.get_test_loader(test_all, test_audio, config)
        model = models.get_model(config)
        model = models.prepare_for_inference(
            model, expdir / f"fold{i}/checkpoints/best.pth").to(device)

        if config["inference"]["prediction_type"] == "strong":
            ##################################################
            # OOF #
            ##################################################
            logger.info("*" * 20)
            logger.info(f"OOF prediction for fold{i}")
            logger.info("*" * 20)
            recording_ids = []
            batch_predictions = []
            indices = []
            for batch in tqdm(val_loader, leave=True):
                recording_ids.extend(batch["recording_id"])
                indices.extend(batch["index"].numpy())
                input_ = batch[global_params["input_key"]].to(device)
                with torch.no_grad():
                    output = model(input_)
                framewise_output = output["framewise_output"].detach()
                clipwise_output, _ = framewise_output.max(dim=1)
                batch_predictions.append(
                    clipwise_output.cpu().numpy())
            oof_prediction = np.concatenate(batch_predictions, axis=0)
            oof_prediction_df = pd.DataFrame(
                oof_prediction, columns=[f"s{i}" for i in range(oof_prediction.shape[1])])
            oof_prediction_df = pd.concat([
                pd.DataFrame({"index": indices, "recording_id": recording_ids}),
                oof_prediction_df
            ], axis=1)

            oof_predictions.append(oof_prediction_df)
            oof_name = "oof_strong.csv"

            ##################################################
            # Prediction #
            ##################################################
            logger.info("*" * 20)
            logger.info(f"Prediction on test for fold{i}")
            logger.info("*" * 20)
            recording_ids = []
            batch_predictions = []
            for batch in tqdm(loader, leave=True):
                recording_ids.extend(batch["recording_id"])
                input_ = batch[global_params["input_key"]].to(device)
                with torch.no_grad():
                    output = model(input_)
                framewise_output = output["framewise_output"].detach()
                clipwise_output, _ = framewise_output.max(dim=1)
                batch_predictions.append(
                    clipwise_output.cpu().numpy())
            fold_prediction = np.concatenate(batch_predictions, axis=0)

            fold_prediction_df = pd.DataFrame(
                fold_prediction, columns=[f"s{i}" for i in range(fold_prediction.shape[1])])
            fold_prediction_df = pd.concat([
                pd.DataFrame({"recording_id": recording_ids}),
                fold_prediction_df
            ], axis=1)

            fold_prediction_df = fold_prediction_df.groupby(
                "recording_id").max().reset_index(drop=False)
            fold_predictions.append(fold_prediction_df)
            submission_name = "strong.csv"
        else:
            ##################################################
            # OOF #
            ##################################################
            logger.info("*" * 20)
            logger.info(f"OOF prediction for fold{i}")
            logger.info("*" * 20)
            recording_ids = []
            batch_predictions = []
            indices = []
            for batch in tqdm(val_loader, leave=True):
                recording_ids.extend(batch["recording_id"])
                indices.extend(batch["index"].numpy())
                input_ = batch[global_params["input_key"]].to(device)
                with torch.no_grad():
                    output = model(input_)
                batch_predictions.append(
                    output["clipwise_output"].cpu().numpy())
            oof_prediction = np.concatenate(batch_predictions, axis=0)
            oof_prediction_df = pd.DataFrame(
                oof_prediction, columns=[f"s{i}" for i in range(oof_prediction.shape[1])])
            oof_prediction_df = pd.concat([
                pd.DataFrame({"index": indices, "recording_id": recording_ids}),
                oof_prediction_df
            ], axis=1)

            oof_predictions.append(oof_prediction_df)
            oof_name = "oof_weak.csv"

            ##################################################
            # Prediction #
            ##################################################
            logger.info("*" * 20)
            logger.info(f"Prediction on test for fold{i}")
            logger.info("*" * 20)
            recording_ids = []
            batch_predictions = []
            for batch in tqdm(loader, leave=True):
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
            fold_predictions.append(fold_prediction_df)
            submission_name = "weak.csv"

    oof_df = pd.concat(oof_predictions, axis=0).reset_index(drop=True)

    oof_df.to_csv(submission_file_dir / oof_name, index=False)

    folds_prediction_df = pd.concat(fold_predictions, axis=0).reset_index(drop=True)
    folds_prediction_df = folds_prediction_df.groupby("recording_id").mean().reset_index(drop=False)

    assert len(folds_prediction_df) == len(submission), \
        "prediction length does not match sample submission length"
    assert folds_prediction_df.shape[1] == submission.shape[1], \
        "number of classes in prediction does not match that of sample submission"
    assert len(set(folds_prediction_df["recording_id"]) - set(submission["recording_id"])) == 0, \
        "recording_id in prediction has unknown value"
    assert len(set(submission["recording_id"]) - set(folds_prediction_df["recording_id"])) == 0, \
        "prediction doesn't have enough recording_id"

    folds_prediction_df.to_csv(submission_file_dir / submission_name, index=False)
