from pathlib import Path
from typing import Dict, List, Tuple, Union

import pytorch_lightning as pl
from torch import Tensor
import torch
from torch.optim import AdamW
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.data import DataLoader

from .dataset import ImagePoseDataset, PathLike, get_dataset_tr, input_shape
from .metric import NavNetLosses
from .model import get_net_by_name


class NavNetPL(pl.LightningModule):
    def __init__(
        self,
        backbone_name: str,
        training_target: str,
        dataset_prefix: PathLike,
        optimizer_args: Dict = None,
        scheduler_args: Dict = None,
        dataloader_args: Dict = None,
    ):
        super().__init__()
        n_outputs = 2 if training_target == "combined" else 1
        self.net = get_net_by_name(backbone_name, n_outputs)
        self.training_target = training_target
        self.metric = NavNetLosses(training_target)

        dataloader_args = dataloader_args or {}
        dataset_prefix = Path(dataset_prefix)
        train_set = ImagePoseDataset(
            dataset_prefix / "train", get_dataset_tr(augment=True)
        )
        val_set = ImagePoseDataset(
            dataset_prefix / "val", get_dataset_tr(augment=False)
        )
        self.train_loader = DataLoader(train_set, **dataloader_args, shuffle=True)
        self.val_loader = DataLoader(val_set, **dataloader_args)

        optimizer_args = optimizer_args or {}
        self.optim = AdamW(self.parameters(), **optimizer_args)
        self.scheduler_args = scheduler_args

        self.save_hyperparameters()

        self._onnx_exporting = False
        self.register_buffer(
            "mean", torch.tensor([0.485, 0.456, 0.406])[None, :, None, None]
        )
        self.register_buffer(
            "std", torch.tensor([0.229, 0.224, 0.225])[None, :, None, None]
        )

    def forward(self, x):
        if self._onnx_exporting:
            x = x.div_(255).sub_(self.mean).div_(self.std)
        return self.net(x)

    def training_step(self, batch: Tuple[Tensor, List[Tensor]], batch_idx: int):
        features, targets = batch
        model_output = self(features)
        loss_dict = self.metric(model_output, targets)
        for k, v in loss_dict.items():
            self.log(f"train/{k}", v)
        return loss_dict[f"{self.training_target}/l2_mean"]

    def validation_step(self, batch: Tuple[Tensor, List[Tensor]], batch_idx: int):
        features, targets = batch
        model_output = self(features)
        loss_dict = self.metric(model_output, targets)
        for k, v in loss_dict.items():
            self.log(f"val/{k}", v)
        return loss_dict[f"{self.training_target}/l1_mean"]

    @property
    def validation_metric(self):
        return f"val/{self.training_target}/l1_mean", "min"

    def configure_optimizers(self):
        if self.scheduler_args is None:
            return self.optim
        val_metric, mode = self.validation_metric
        scheduler = ReduceLROnPlateau(self.optim, **self.scheduler_args, mode=mode)
        return {
            "optimizer": self.optim,
            "lr_scheduler": scheduler,
            "monitor": val_metric,
        }

    def train_dataloader(self):
        return self.train_loader

    def val_dataloader(self):
        return self.val_loader

    def test_step(self, batch: Tuple[Tensor, List[Tensor]], batch_idx: int):
        return self.validation_step(batch, batch_idx)
    
    def test_dataloader(self):
        return self.val_loader

    @property
    def example_input_array(self):
        return torch.zeros(*input_shape)

    def to_onnx(self, file_path: Union[str, Path], input_sample = None, **kwargs):
        self._onnx_exporting = True
        ret = super().to_onnx(file_path, input_sample=input_sample, **kwargs)
        self._onnx_exporting = False
        return ret
