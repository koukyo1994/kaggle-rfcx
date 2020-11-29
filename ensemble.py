import numpy as np
import pandas as pd

import datasets
import utils

from pathlib import Path

from tqdm import tqdm

from sklearn.metrics import log_loss

from callbacks import lwlrap


def search_averaging_weights(predictions: list, target: np.ndarray, trials=1000):
    best_loss = np.inf
    best_weights = np.zeros(len(predictions))
    utils.set_seed(1213)

    for i in tqdm(range(trials)):
        dice = np.random.rand(len(predictions))
        weights = dice / dice.sum()
        blended = np.zeros(len(predictions[0]))
        for weight, pred in zip(weights, predictions):
            blended += weight * pred
        loss = log_loss(y_true=target, y_pred=blended)
        if loss < best_loss:
            best_loss = loss
            best_weights = weights
    return {"best_loss": best_loss, "best_weights": best_weights}


if __name__ == "__main__":
    args = utils.get_parser().parse_args()
    config = utils.load_config(args.config)

    config_name = args.config.split("/")[-1].replace(".yml", "")
    expdir = Path(f"out/{config_name}")
    expdir.mkdir(exist_ok=True, parents=True)

    logger = utils.get_logger(expdir / "ensemble.log")

    oofs = []
    submissions = []
    for result_dict in config["results"]:
        oofs.append(pd.read_csv(result_dict["oof"]))
        submissions.append(pd.read_csv(result_dict["submission"]))

    tp, _, _, _, _, _ = datasets.get_metadata(config)
    indices = tp[["index"]]

    for i in range(len(oofs)):
        oofs[i] = indices.merge(oofs[i], on="index", how="left")

    labels = []
    for _, sample in tp.iterrows():
        t_min = sample["t_min"]
        t_max = sample["t_max"]
        flac_id = sample["recording_id"]
        call_duration = t_max - t_min
        relative_offset = (10 - call_duration) / 2

        offset = min(max(0, t_min - relative_offset), 60 - 10)
        tail = offset + 10

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

    weights_dict = {}
    classes = [f"s{i}" for i in range(24)]
    for class_ in classes:
        logger.info("*" * 20)
        logger.info(f"class: {class_}")

        predictions = []
        for oof in oofs:
            predictions.append(oof[class_].values)
        target = ground_truth_df[class_].values
        result_dict = search_averaging_weights(predictions, target)

        logger.info(
            f"Best Loss: {result_dict['best_loss']}, Best Weights{result_dict['best_weights']}")
        weights_dict[class_] = result_dict["best_weights"]

    blended = np.zeros((len(oofs[0]), 24))
    for class_ in weights_dict:
        index = classes.index(class_)
        for weight, oof in zip(weights_dict[class_], oofs):
            blended[index] += weight * oof[class_].values

    score_class, weight = lwlrap(ground_truth_df[classes].values, blended)
    score = (score_class * weight).sum()
    logger.info(f"Blended LWLRAP: {score:5f}")

    blended_sub = np.zeros((len(submissions[0]), 24))
    for class_ in weights_dict:
        index = classes.index(class_)
        for weight, sub in zip(weights_dict[class_], submissions):
            blended_sub[index] += weight * sub[class_].values

    sub = pd.concat([
        pd.DataFrame({"recording_id": submissions[0]["recording_id"]}),
        pd.DataFrame(blended_sub, columns=classes)
    ], axis=1)
    sub.to_csv(expdir / "blended.csv", index=False)
