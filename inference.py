import sys
import warnings

import numpy as np
import pandas as pd
import torch

import datasets
import models
import training
import utils

from pathlib import Path

from tqdm import tqdm

from callbacks import lwlrap


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

    # labels
    labels = []
    duration = config["dataset"]["valid"]["params"]["duration"]
    for _, sample in tp.iterrows():
        t_min = sample["t_min"]
        t_max = sample["t_max"]
        flac_id = sample["recording_id"]
        call_duration = t_max - t_min
        relative_offset = (duration - call_duration) / 2

        offset = min(max(0, t_min - relative_offset), 60 - duration)
        tail = offset + duration

        query_string = f"recording_id == '{flac_id}' & "
        query_string += f"t_min < {tail} & t_max > {offset}"
        all_tp_events = tp.query(query_string)

        label = np.zeros(24, dtype=np.float32)
        for species_id in all_tp_events["species_id"].unique():
            label[int(species_id)] = 1.0
        labels.append(label)

    labels_df = pd.DataFrame(np.asarray(labels), columns=[f"s{i}" for i in range(24)])
    ground_truth_df = pd.concat([
        pd.DataFrame({"index": tp["index"], "recording_id": tp["recording_id"]}),
        labels_df
    ], axis=1)

    # soft prediction
    if config["inference"].get("soft_prediction", False):
        soft_inference_config = {
            "dataset": {
                "valid": {
                    "name": "SampleWiseSpectrogramDataset",
                    "params": config["dataset"]["valid"]["params"]
                },
                "test": {
                    "name": "SampleWiseSpectrogramTestDataset",
                    "params": config["dataset"]["test"]["params"]
                }
            },
            "loader": {
                "valid": {
                    "batch_size": 1,
                    "shuffle": False,
                    "num_workers": 20
                },
                "test": {
                    "batch_size": 1,
                    "shuffle": False,
                    "num_workers": 20
                }
            }
        }
        soft_test_loader = datasets.get_test_loader(
            test_all, test_audio, soft_inference_config)

        soft_oof_dir = expdir / "soft_oof"
        soft_oof_dir.mkdir(exist_ok=True, parents=True)

        soft_prediction_dir = expdir / "soft_prediction"
        soft_prediction_dir.mkdir(exist_ok=True, parents=True)

        soft_predictions = {}

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
        if config["inference"].get("last", False):
            model = models.prepare_for_inference(
                model, expdir / f"fold{i}/checkpoints/last.pth").to(device)
            last = True
        else:
            model = models.prepare_for_inference(
                model, expdir / f"fold{i}/checkpoints/best.pth").to(device)
            last = False

        if config["inference"].get("soft_prediction", False):
            soft_val_loader = datasets.get_train_loader(
                val_df, tp, fp, train_audio, soft_inference_config, phase="valid")

            logger.info("*" * 20)
            logger.info(f"Soft OOF prediction for fold{i}")
            logger.info("*" * 20)
            for batch in tqdm(soft_val_loader, desc="soft oof"):
                input_ = batch[global_params["input_key"]].squeeze(0).to(device)
                with torch.no_grad():
                    output = model(input_)
                framewise_output = output["framewise_output"].detach().cpu().numpy()
                clip_prediction = np.vstack(framewise_output).astype(np.float16)
                recording_id = batch["recording_id"][0]
                np.savez_compressed(soft_oof_dir / recording_id, clip_prediction)

            logger.info("*" * 20)
            logger.info("Soft prediction for test")
            logger.info("*" * 20)
            for batch in tqdm(soft_test_loader, desc="soft test"):
                input_ = batch[global_params["input_key"]].squeeze(0).to(device)
                with torch.no_grad():
                    output = model(input_)
                framewise_output = output["framewise_output"].detach().cpu().numpy()
                clip_prediction = np.vstack(framewise_output).astype(np.float16)
                recording_id = batch["recording_id"][0]
                soft_predictions[recording_id] = clip_prediction / len(global_params["folds"])

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
            if last:
                oof_name = "oof_strong_last.csv"
            else:
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
            if last:
                submission_name = "strong_last.csv"
            else:
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
            if last:
                oof_name = "oof_weak_last.csv"
            else:
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
            if last:
                submission_name = "weak_last.csv"
            else:
                submission_name = "weak.csv"

    oof_df = pd.concat(oof_predictions, axis=0).reset_index(drop=True)

    oof_indices = oof_df[["index"]]
    ground_truth = oof_indices.merge(ground_truth_df, on="index", how="left")
    columns = [f"s{i}" for i in range(24)]
    score_class, weight = lwlrap(ground_truth[columns].values, oof_df[columns].values)
    score = (score_class * weight).sum()
    logger.info(f"Valid LWLRAP: {score:.5f}")
    class_level_score = {config_name: score_class}
    utils.save_json(class_level_score, submission_file_dir / "class_level_results.json")

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

    if config["inference"].get("soft_prediction", False):
        for key in soft_predictions:
            np.savez_compressed(soft_prediction_dir / key, soft_predictions[key])
