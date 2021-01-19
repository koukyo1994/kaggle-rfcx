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

    logger = utils.get_logger(expdir / "tta.log")

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

    # validation
    splitter = training.get_split(config)

    ##################################################
    # Main Loop #
    ##################################################
    fold_predictions = []
    oof_predictions = []
    oof_tta_dict: dict = {}
    for tta_conf in config["tta"]:
        oof_tta_dict[tta_conf["name"]] = []
    for i, (trn_idx, val_idx) in enumerate(splitter.split(train_all)):
        if i not in global_params["folds"]:
            continue
        logger.info("=" * 20)
        logger.info(f"Fold {i}")
        logger.info("=" * 20)

        val_df = train_all.iloc[val_idx, :].reset_index(drop=True)
        model = models.get_model(config)
        if config["inference"].get("last", False):
            model = models.prepare_for_inference(
                model, expdir / f"fold{i}/checkpoints/last.pth").to(device)
            last = True
        else:
            model = models.prepare_for_inference(
                model, expdir / f"fold{i}/checkpoints/best.pth").to(device)
            last = False

        ttas = config["tta"]
        oof_tta_predictions = []
        tta_predictions = []
        for tta in ttas:
            logger.info("#" * 20)
            logger.info(tta["name"])

            _config = config.copy()
            _config["transforms"]["valid"] = [tta]
            val_loader = datasets.get_train_loader(
                val_df, tp, fp, train_audio, _config, phase="valid")

            _config["transforms"]["test"] = [tta]

            loader = datasets.get_test_loader(test_all, test_audio, _config)

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

                oof_tta_predictions.append(oof_prediction_df)
                oof_tta_dict[tta["name"]].append(oof_prediction_df)
                if last:
                    oof_name = "oof_tta_strong_last.csv"
                else:
                    oof_name = "oof_tta_strong.csv"

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
                tta_predictions.append(fold_prediction_df)
                if last:
                    submission_name = "tta_strong_last.csv"
                else:
                    submission_name = "tta_strong.csv"
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

                oof_tta_predictions.append(oof_prediction_df)
                oof_tta_dict[tta["name"]].append(oof_prediction_df)
                if last:
                    oof_name = "oof_tta_weak_last.csv"
                else:
                    oof_name = "oof_tta_weak.csv"

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
                tta_predictions.append(fold_prediction_df)
                if last:
                    submission_name = "tta_weak_last.csv"
                else:
                    submission_name = "tta_weak.csv"
        oof_tta_df = pd.concat(oof_tta_predictions, axis=0).reset_index(drop=True)
        oof_tta_df = oof_tta_df.groupby(["index", "recording_id"])[[f"s{i}" for i in range(24)]].mean().reset_index(drop=False)
        oof_predictions.append(oof_tta_df)

        tta_pred_df = pd.concat(tta_predictions, axis=0).reset_index(drop=True)
        tta_pred_df = tta_pred_df.groupby(["recording_id"])[[f"s{i}" for i in range(24)]].mean().reset_index(drop=False)
        fold_predictions.append(tta_pred_df)

    oof_df = pd.concat(oof_predictions, axis=0).reset_index(drop=True)

    oof_indices = oof_df[["index"]]
    ground_truth = oof_indices.merge(ground_truth_df, on="index", how="left")
    columns = [f"s{i}" for i in range(24)]
    score_class, weight = lwlrap(ground_truth[columns].values, oof_df[columns].values)
    score = (score_class * weight).sum()
    logger.info(f"TTA all LWLRAP: {score:.5f}")

    class_level_score = {config_name: score_class}

    for key in oof_tta_dict:
        tta_df = pd.concat(oof_tta_dict[key], axis=0).reset_index(drop=True)
        oof_indices = tta_df[["index"]]
        ground_truth = oof_indices.merge(ground_truth_df, on="index", how="left")
        score_class, weight = lwlrap(ground_truth[columns].values, tta_df[columns].values)
        score = (score_class * weight).sum()
        logger.info(f"TTA {key} LWLRAP: {score:.5f}")
        class_level_score[key] = score_class

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
