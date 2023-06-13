import logging

import pytorch_lightning as pl

from approxcaliper import compress, init_logging

logger = logging.getLogger(__name__)
pl.seed_everything(1112)


def read_config():
    import importlib
    from argparse import ArgumentParser
    from pathlib import Path
    from warnings import warn

    import yaml

    TASKS = {"train": train, "prune": prune, "lr-fac": lr_fac}

    parser = ArgumentParser(description="Train/prune a pytorch-lightning network")
    parser.add_argument(
        "-c", "--checkpoint", type=str, help="Checkpoint to resume training from"
    )
    parser.add_argument("config_file", type=Path, help="Path to configuration file")
    parser.add_argument(
        "config_name", type=str, help="Name of the config in config file to run"
    )
    args = parser.parse_args()
    with open(args.config_file) as f:
        config = yaml.safe_load(f)
    config: dict = config[args.config_name]

    func = TASKS[config.pop("__task__")]
    # Init network
    dnn_conf = config.pop("__dnn__")
    dnn_package, dnn_cls = dnn_conf["__cls__"].rsplit(".", 1)
    dnn_cls = getattr(importlib.import_module(dnn_package), dnn_cls)
    if args.checkpoint:
        dnn = dnn_cls.load_from_checkpoint(args.checkpoint, **dnn_conf)
    elif not args.checkpoint and func != train:
        warn(
            "Iterative pruning should start with trained model; please provide a checkpoint"
        )
        dnn = dnn_cls(**dnn_conf)
    else:
        dnn = dnn_cls(**dnn_conf)
    if not isinstance(dnn, pl.LightningModule):
        raise ValueError("Network must be a LightningModule")
    return config, dnn, func


def train(dnn: pl.LightningModule, config: dict):
    trainer = make_trainer(dnn, config)
    # Initialize logger here so we can put log file
    # to the same directory as the trainer saves things
    init_logging(trainer)
    logger.info(config)
    # Start training
    trainer.fit(dnn)


def prune(dnn: pl.LightningModule, config: dict):
    val_metric, mode = dnn.val_metric
    helper = compress.PLModelHelper(
        lambda: make_trainer(val_metric, mode, config, False), val_metric
    )
    init_logging(helper.log_dir)
    logger.info(config)
    compress.iterative_prune_dnn(
        dnn,
        helper,
        config["prune_scheduler"],
        config["pruner"],
        config["prune_ratio"],
        config["n_prune_steps"],
        keep_intermediate_result=True,
        speedup=True,
    )
    # task_id, pruned_model, masks, score, configs = pruner.get_best_result()


def lr_fac(dnn: pl.LightningModule, config: dict):
    val_metric, mode = dnn.val_metric
    helper = compress.PLModelHelper(
        lambda: make_trainer(val_metric, mode, config, False), val_metric
    )
    init_logging(helper.log_dir)
    logger.info(config)
    compress.iterative_lr_compress_dnn(
        dnn, helper, config["final_sparsity"], config["n_lr_steps"]
    )


def make_trainer(val_metric: str, mode: str, config: dict, save_model: bool = True):
    from pytorch_lightning import callbacks as plcb

    lrm = plcb.LearningRateMonitor()
    if save_model:
        filename = "epoch={epoch}-metric={%s:.3f}" % (val_metric,)
        ckpt = plcb.ModelCheckpoint(
            monitor=val_metric,
            filename=filename,
            mode=mode,
            save_top_k=3,
            auto_insert_metric_name=False,
        )
        callbacks = [ckpt, lrm]
    else:
        callbacks = [lrm]
    early_stop = config.get("early_stop")
    if early_stop is not None:
        earlystop_cb = plcb.EarlyStopping(
            monitor=val_metric, mode=mode, verbose=True, **early_stop
        )
        callbacks.append(earlystop_cb)
    return pl.Trainer(
        gpus=config["gpus"],
        accelerator="dp",
        max_epochs=config["train_epochs"],
        callbacks=callbacks,
        checkpoint_callback=save_model,
    )


if __name__ == "__main__":
    config_, dnn, task_handler = read_config()
    task_handler(dnn, config_)
