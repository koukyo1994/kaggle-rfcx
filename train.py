import callbacks as clb
import criterions
import datasets
import models
import training
import utils

from pathlib import Path

from catalyst.dl import SupervisedRunner


if __name__ == "__main__":
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

        trn_df = train_all.loc[trn_idx, :].reset_index(drop=True)
        val_df = train_all.loc[val_idx, :].reset_index(drop=True)

        loaders = {
            phase: datasets.get_train_loader(df_, tp, fp, train_audio, config, phase)
            for df_, phase in zip([trn_df, val_df], ["train", "valid"])
        }
        model = models.get_model(config).to(device)
        criterion = criterions.get_criterion(config)
        optimizer = training.get_optimizer(model, config)
        scheduler = training.get_scheduler(optimizer, config)
        callbacks = clb.get_callbacks(config)

        runner = SupervisedRunner(
            device=device,
            input_key=global_params["input_key"],
            input_target_key=global_params["input_target_key"])
        runner.train(
            model=model,
            criterion=criterion,
            loaders=loaders,
            optimizer=optimizer,
            scheduler=scheduler,
            num_epochs=global_params["num_epochs"],
            verbose=True,
            logdir=logdir / f"fold{i}",
            callbacks=callbacks,
            main_metric=global_params["main_metric"],
            minimize_metric=global_params["minimize_metric"])
