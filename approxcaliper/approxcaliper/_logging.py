import logging
import os
import time
from pathlib import Path
from typing import Union

import opentuner
from pytorch_lightning import Trainer
from tqdm import tqdm

logger = logging.getLogger(__name__)
__all__ = ["init_logging"]


class TqdmStreamHandler(logging.Handler):
    """tqdm-friendly logging handler. Uses tqdm.write instead of print for logging."""

    def __init__(self, level=logging.NOTSET):
        super().__init__(level)

    def emit(self, record):
        try:
            msg = self.format(record)
            tqdm.write(msg)
            self.flush()
        except (KeyboardInterrupt, SystemExit, RecursionError):
            raise
        except:
            self.handleError(record)


def init_logging(dir_or_trainer: Union[None, str, Path, Trainer]):
    output_dir: Path
    if dir_or_trainer is None:
        output_dir = Path(".")
    elif isinstance(dir_or_trainer, Trainer):
        output_dir = resolve_trainer_dir(dir_or_trainer)
    else:
        output_dir = Path(dir_or_trainer)
    timestr = time.strftime("%Y.%m.%d-%H%M%S.log")
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    file_path = output_dir / timestr

    # Evil hack, changes the logging configuration of opentuner
    opentuner.tuningrunmain.the_logging_config = {
        "version": 1,
        "disable_existing_loggers": False,
        "formatters": {
            "console": {
                "format": "[%(relativeCreated)6.0fs] "
                "%(levelname)7s %(name)s: "
                "%(message)s"
            },
            "file": {
                "format": "[%(asctime)-15s] "
                "%(levelname)7s %(name)s: "
                "%(message)s "
                "@%(filename)s:%(lineno)d"
            },
        },
        "handlers": {
            "console": {
                "()": TqdmStreamHandler,
                "formatter": "console",
                "level": "INFO",
            },
            "file": {
                "class": "logging.FileHandler",
                "filename": file_path.as_posix(),
                "formatter": "file",
                "level": "DEBUG",
            },
        },
        "loggers": {
            "": {"handlers": ["console", "file"], "level": "DEBUG", "propagate": True}
        },
    }
    opentuner.init_logging()
    logger.info(f"Logging to {file_path}")
    logger_names = [
        "nni.compression.pytorch.speedup.compressor",
        "nni.compression.pytorch.speedup.compress_modules",
        "FixMaskConflict",
    ]
    for name in logger_names:
        logging.getLogger(name).setLevel(logging.WARNING)


def resolve_trainer_dir(trainer: Trainer):
    from pytorch_lightning.loggers import TensorBoardLogger

    if not isinstance(trainer.logger, TensorBoardLogger):
        return Path(trainer.default_root_dir)
    return Path(trainer.logger.log_dir)
