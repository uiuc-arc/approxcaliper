import logging
from copy import deepcopy
from pathlib import Path
from typing import Callable, Dict, List, Optional, Tuple, Union

import pytorch_lightning as pl
import torch
from nni.algorithms.compression.v2.pytorch import base
from nni.algorithms.compression.v2.pytorch import pruning as pr
from nni.algorithms.compression.v2.pytorch import utils
from nni.compression.pytorch.utils.counter import count_flops_params
from torch import Tensor
from torch.nn import Conv2d, ConvTranspose2d, Module, Sequential

logger = logging.getLogger(__name__)
__all__ = ["iterative_prune_dnn"]


def iterative_prune_dnn(
    model: pl.LightningModule,
    pl_helper: "PLModelHelper",
    prune_scheduler: str,
    pruner: str,
    prune_ratio: float,
    total_iteration: int,
    **pruner_kwargs,
):
    args = model.transfer_batch_to_device(model.example_input_array)
    flops0, params0, _ = count_flops_params(model, args, verbose=False)
    logger.info("Baseline FLOPs = %d, params = %d", flops0, params0)
    PRUNERS = {
        "linear": pr.LinearPruner,
        "agp": pr.AGPPruner,
        "lottery": pr.LotteryTicketPruner,
    }
    if prune_scheduler not in PRUNERS:
        raise ValueError(f"Unknown prune_scheduler: {prune_scheduler}")
    config_list = [{"total_sparsity": prune_ratio, "op_types": ["Conv2d"]}]
    if pruner == "lr_reg":
        pruner_kwargs["reset_weight"] = False
    prune_scheduler_inst: pr.PruningScheduler = PRUNERS[prune_scheduler](
        model=model,
        config_list=config_list,
        pruning_algorithm=pruner,
        total_iteration=total_iteration,
        dummy_input=args,
        finetuner=pl_helper.finetuner,
        evaluator=pl_helper.evaluator,
        log_dir=pl_helper.log_dir,
        **pruner_kwargs,
    )
    prune_scheduler_inst.compress()


def iterative_lr_compress_dnn(
    model: pl.LightningModule,
    pl_helper: "PLModelHelper",
    lr_ratio: float,
    total_iteration: int,
    **pruner_kwargs,
):
    args = model.transfer_batch_to_device(model.example_input_array)
    flops0, params0, _ = count_flops_params(model, args, verbose=False)
    logger.info("Baseline FLOPs = %d, params = %d", flops0, params0)
    config_list = [
        {"total_sparsity": lr_ratio, "op_types": ["Conv2d", "ConvTranspose2d"]}
    ]
    lr_scheduler = LRScheduler(
        model=model,
        config_list=config_list,
        total_iteration=total_iteration,
        dummy_input=args,
        finetuner=pl_helper.finetuner,
        evaluator=pl_helper.evaluator,
        log_dir=pl_helper.log_dir,
        **pruner_kwargs,
    )
    lr_scheduler.compress()


class PLModelHelper:
    def __init__(self, trainer_ctor: Callable[[], pl.Trainer], val_metric: str) -> None:
        self.trainer_ctor = trainer_ctor
        self.val_metric = val_metric
        # Create a temp trainer to see its log directory
        trainer = self.trainer_ctor()
        self.log_dir = Path(trainer.logger.log_dir)
        self._iteration = 0

    def finetuner(self, pruned_model: Module):
        assert isinstance(
            pruned_model, pl.LightningModule
        ), "`module` must be a pytorch-lightning module"
        self._iteration += 1
        dummy_input = pruned_model.transfer_batch_to_device(
            pruned_model.example_input_array
        )
        flops, params, _ = count_flops_params(pruned_model, dummy_input, verbose=False)
        logger.info("FLOPs = %d, params = %d", flops, params)
        trainer = self._make_trainer(log=True)
        trainer.fit(pruned_model)

    def evaluator(self, pruned_model: Module):
        assert isinstance(
            pruned_model, pl.LightningModule
        ), "`module` must be a pytorch-lightning module"
        trainer = self._make_trainer(log=False)
        result = trainer.validate(pruned_model)[0]
        logger.info("metrics: %s", result)
        return result[self.val_metric]

    def _make_trainer(self, log: bool):
        from pytorch_lightning.loggers import TensorBoardLogger

        trainer = self.trainer_ctor()
        if log:
            trainer.logger = TensorBoardLogger(
                save_dir=self.log_dir.parent.as_posix(),
                name=self.log_dir.name,
                version=f"iteration_{self._iteration}",
            )
        else:
            trainer.logger = None
        return trainer


class LRWrapper(Module):
    def __init__(
        self, module: Union[Conv2d, ConvTranspose2d], name: str, config: dict
    ) -> None:
        super().__init__()
        self.name = name
        self.config = config
        self._orig_module_str = str(module)
        self._orig_pad = module.padding
        self._orig_stride = module.stride
        self.module = module
        if isinstance(module, Conv2d):
            self.conv_cls = Conv2d
            self.forward_permute = 1, 2, 3, 0
            self.backward_permute = 3, 0, 1, 2
        elif isinstance(module, ConvTranspose2d):
            self.conv_cls = ConvTranspose2d
            self.forward_permute = 0, 2, 3, 1
            self.backward_permute = 0, 3, 1, 2
        else:
            raise ValueError("Unknown module type")

    def compress(self, lr_rate: float):
        from math import ceil

        W, B = self._get_weight()
        ci, d0, d1, co = W.shape
        assert d0 == d1, "Only support square convolution"
        rank0 = min(co * d0, ci * d0)
        rank = ceil(rank0 * lr_rate)
        flops0, flops = d0 * d0 * co * ci, d0 * rank * (co + ci)
        if rank0 == rank or flops > flops0:
            return
        # SVD approximation
        # Adapted from https://github.com/chengtaipu/lowrankcnn/blob/master/imagenet/lowrank_approx.py
        W_ = W.reshape(ci * d0, d0 * co)
        U, S, V = torch.linalg.svd(W_, full_matrices=False)
        Ssqrt = torch.sqrt(S)
        v: Tensor = U[:, :rank] * Ssqrt[:rank]
        h: Tensor = V[:rank, :] * Ssqrt[:rank, None]
        v = v.reshape(ci, d0, 1, rank)
        h = h.reshape(rank, 1, d0, co)
        self.module = self._make_layers(v, h, B)

    def forward(self, x: Tensor):
        return self.module(x)

    def _make_layers(self, w1: Tensor, w2: Tensor, b: Optional[Tensor]) -> Module:
        permute = self.backward_permute
        cls = self.conv_cls
        pad_h, pad_w = self._orig_pad
        stride_h, stride_w = self._orig_stride
        assert isinstance(pad_h, int) and isinstance(pad_w, int)
        ci, d0, _, rank = w1.shape
        co = w2.shape[-1]
        conv_v = cls(ci, rank, (d0, 1), (stride_h, 1), (pad_h, 0), bias=False)
        conv_v.weight.data = w1.permute(permute)
        conv_h = cls(rank, co, (1, d0), (1, stride_w), (0, pad_w), bias=b is not None)
        conv_h.weight.data = w2.permute(permute)
        if b is not None:
            assert isinstance(conv_h.bias, Tensor)
            conv_h.bias.data = b.data
        return Sequential(conv_v, conv_h)

    def _get_weight(self) -> Tuple[Tensor, Optional[Tensor]]:
        permute = self.forward_permute
        cls = self.conv_cls
        if isinstance(self.module, cls):
            M = self.module
            assert M.groups == 1 and M.dilation == (1, 1)
            # I, O, H, W -> I, H, W, O (ci, d0, d0, co)
            return M.weight.permute(permute), M.bias
        conv_v, conv_h = self.module
        assert isinstance(conv_v, cls) and isinstance(conv_h, cls)
        # ci, rank, h, 1 -> ci, d0, 1, rank
        V = conv_v.weight.permute(permute)
        ci, d0, _, rank = V.shape
        V = V.reshape(ci * d0, rank)
        # rank, co, 1, w -> rank, 1, d0, co
        H = conv_h.weight.permute(permute)
        co = H.shape[3]
        H = H.reshape(rank, d0 * co)
        # ci, d0, d0, co
        W = (V @ H).reshape(ci, d0, d0, co)
        return W, conv_h.bias


class LRReg(base.Compressor):
    def __init__(self, model: Module):
        super().__init__(None, None)
        self.bound_model = model

    def compress(self, config_list: List[Dict]) -> Module:
        assert self.bound_model is not None
        if self.is_wrapped:
            self.config_list = config_list
            self.validate_config(model=self.bound_model, config_list=config_list)
        else:
            self.reset(self.bound_model, config_list)
        assert self._modules_to_compress is not None

        wrappers = self.get_modules_wrapper()
        for layer, _ in self._modules_to_compress:
            config = self._select_config(layer)
            wrapper = wrappers[layer.name]
            assert isinstance(wrapper, LRWrapper) and config is not None
            wrapper.compress(1 - config["total_sparsity"])
        return self.bound_model

    def show_pruned_weights(self, dim: int = 0):
        for wrapper in self.get_modules_wrapper().values():
            assert isinstance(wrapper, LRWrapper)
            logger.info(
                f"LR regularization {wrapper.name}: \n"
                f"  {wrapper._orig_module_str} -> {wrapper.module}"
            )

    def _wrap_modules(self, layer: base.LayerInfo, config: Dict):
        assert isinstance(layer.module, (Conv2d, ConvTranspose2d))
        logger.debug("Wrapping module %s with wrapper", layer.name)
        wrapper = LRWrapper(layer.module, layer.name, config)
        layer_weight = getattr(layer.module, "weight", None)
        assert isinstance(
            layer_weight, Tensor
        ), f"module {layer.name} does not have a Tensor weight"
        wrapper.to(layer_weight.device)
        return wrapper


class LRScheduler:
    def __init__(
        self,
        model: Module,
        config_list: List[Dict],
        total_iteration: int,
        log_dir: Path,
        finetuner: Callable[[Module], None],
        evaluator: Callable[[Module], float],
        dummy_input: Tensor,
    ):
        self.total_iteration = total_iteration
        self.lr_compressor = LRReg(model)
        self.finetuner = finetuner
        self.evaluator = evaluator
        self.dummy_input = dummy_input

        self.target_sparsity = utils.config_list_canonical(model, config_list)
        self._baseline_flops, _, _ = count_flops_params(
            model, self.dummy_input, verbose=False
        )
        self._model_dir = Path(log_dir, "models").absolute()
        self._model_dir.mkdir(parents=True, exist_ok=True)

    def compress(self):
        import gc

        # Export onnx model for baseline
        self.save_model(0)
        # Compress model
        for iteration in range(1, self.total_iteration + 1):
            config_list = []
            actual_sparsity = 0.0
            for target in self.target_sparsity:
                original = 1 - (1 - target["total_sparsity"]) ** (
                    iteration / self.total_iteration
                )
                sparsity = max(
                    0.0, (original - actual_sparsity) / (1 - actual_sparsity)
                )
                assert 0 <= sparsity <= 1
                config_list.append(deepcopy(target))
                config_list[-1]["total_sparsity"] = sparsity
                logger.info("Sparsity %.4f", sparsity)
            compact_model = self.pruning_one_step(config_list)
            actual_sparsity = self.get_actual_sparsity(compact_model)
            assert compact_model == self.lr_compressor.bound_model
            self.save_model(iteration)
            gc.collect()
        self.lr_compressor._unwrap_model()

    def pruning_one_step(self, config_list):
        compact_model = self.lr_compressor.compress(config_list)
        # show the pruning effect
        self.lr_compressor.show_pruned_weights()
        # finetune
        self.finetuner(compact_model)
        # evaluate
        score = self.evaluator(compact_model)
        return compact_model

    def save_model(self, iteration: int):
        model_path = self._model_dir / f"{iteration}_model.pth"
        compressor = deepcopy(self.lr_compressor)
        compressor._unwrap_model()
        model = compressor.bound_model
        torch.save(model, model_path)
        if isinstance(model, pl.LightningModule):
            model.to_onnx(model_path.with_suffix(".onnx"))

    def get_actual_sparsity(self, compact_model: Module):
        flops, _, _ = count_flops_params(compact_model, self.dummy_input, verbose=False)
        return flops / self._baseline_flops
