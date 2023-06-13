import abc
import logging
from typing import Any, Callable, List, Tuple, TypeVar

import pytorch_lightning as pl
import torch
from opentuner import MeasurementInterface
from tqdm import tqdm

logger = logging.getLogger(__name__)
T = TypeVar("T")


class ErrorInjector(abc.ABC):
    @property
    @abc.abstractmethod
    def params(self) -> List[Tuple[str, float, float]]:
        pass

    @abc.abstractmethod
    def inject_error(self, prediction: T, **kwargs) -> T:
        pass


class PLErrorTuner:
    def __init__(
        self,
        network: pl.LightningModule,
        error_injector: ErrorInjector,
        metric_function: Callable[[Any, Any], float],
        sampling_schedule: List[Tuple[int, float]],
        n_seeds: int,
    ) -> None:
        from itertools import cycle

        from opentuner import default_argparser

        self.network = network
        self.error_injector = error_injector
        self.metric_function = metric_function
        self.sampling_sched = sampling_schedule
        self.n_seeds = n_seeds

        self.args = args = default_argparser().parse_args([])
        args.no_dups = True  # Don't print duplicated config warnings
        args.parallelism = 1
        self._loop_val_loader = cycle(self.network.val_dataloader())
        self._configs_table = MultiSizeResultTable()

    def __call__(self, metric_val: float):
        from opentuner.tuningrunmain import TuningRunMain

        # Throwing in a float() here the input may be float64, etc.
        # which can confuse opentuner.
        metric_val = float(metric_val)
        logger.info(f"Tuning target: {metric_val}")
        best_conf = None
        for sample_size, eps in self.sampling_sched:
            logger.info(f"Tuning at sample size {sample_size}")
            seeds = self._configs_table.get_seeds(self.n_seeds, metric_val)
            logger.info(f"Using seeds: {seeds}")
            tuner = PLErrorInterface(self, seeds, sample_size, eps, metric_val)
            trm = TuningRunMain(tuner, self.args)
            tuner.set_progress_getter(lambda: trm.search_driver.test_count)
            trm.main()
            self._configs_table.insert(sample_size, tuner.configs)
            best_conf = tuner.configs[0][0]
            logger.info(f"Best config: {best_conf}")
        assert best_conf is not None
        logger.info(f"Final config: {best_conf}")
        metric = self.run_with_error_inj(best_conf, n=-1)
        logger.info(f"Metric value on full dataset: {metric}")
        return best_conf, metric

    @torch.no_grad()
    def run_with_error_inj(self, cfg: dict, n: int):
        from itertools import islice

        from pytorch_lightning.utilities.apply_func import move_data_to_device

        metric_values = 0.0
        val_loader = self.network.val_dataloader()
        batch_size: int = val_loader.batch_size
        if n == -1:
            n = len(val_loader) * batch_size
            loader = val_loader
        else:
            loader = islice(self._loop_val_loader, n // batch_size)
        pbar = tqdm(total=n, leave=False)
        for images, labels in loader:
            images, labels = move_data_to_device((images, labels), self.network.device)
            prediction = self.network(images)
            prediction = self.error_injector.inject_error(prediction, **cfg)
            metric_values += self.metric_function(prediction, labels) * len(images)
            pbar.update(len(images))
        # float() to convert float64, etc. to actual python float.
        return float(metric_values / pbar.n)


class MultiSizeResultTable:
    def __init__(self):
        self.results = []

    def insert(self, size: int, results: List[Tuple[dict, float]]):
        self.results.extend([(conf, size, metric) for conf, metric in results])

    def get_seeds(self, take_n: int, target: float):
        from math import sqrt

        # Sort results with a heuristic formula:
        results = sorted(self.results, key=lambda x: abs(x[2] - target) / sqrt(x[1]))
        return [conf for conf, _, _ in results[:take_n]]


class PLErrorInterface(MeasurementInterface):
    def __init__(
        self,
        tuner: PLErrorTuner,
        seed_configs: List[dict],
        sample_size: int,
        eps: float,
        metric_target: float,
    ):
        from opentuner.measurement.inputmanager import FixedInputManager
        from opentuner.search.objective import MinimizeTime

        self.tuner = tuner
        self.eps = eps
        logger.info(f"Tuning till metric value difference is within {self.eps}")
        self.seeds = seed_configs
        self.sample_size = sample_size
        self.metric_target = metric_target

        self.configs: List[Tuple[dict, float]] = []
        self._pbar = tqdm(leave=False)
        self._progress_getter = None

        # We use ThresholdAccuracyMinimizeTime to threshold metric value from one side
        # while minimizing the difference from the other size.
        # This should be better than just minimizing the distance to the goal.
        objective = MinimizeTime()
        input_manager = FixedInputManager(size=len(tuner.error_injector.params))
        super(PLErrorInterface, self).__init__(
            tuner.args, input_manager=input_manager, objective=objective
        )

    def set_progress_getter(self, getter: Callable[[], int]):
        self._progress_getter = getter

    def seed_configurations(self):
        return self.seeds

    def manipulator(self):
        """Define the search space by creating a ConfigurationManipulator"""
        from opentuner import ConfigurationManipulator, FloatParameter

        manipulator = ConfigurationManipulator()
        for param_name, vmin, vmax in self.tuner.error_injector.params:
            manipulator.add_parameter(FloatParameter(param_name, vmin, vmax))
        return manipulator

    def extra_convergence_criteria(self, results):
        return any(result.time <= self.eps for result in results)

    def run(self, desired_result, input, limit):
        from opentuner.resultsdb.models import Result

        cfg = desired_result.configuration.data
        logger.debug(f"Iteration # {self._pbar.n}: config={cfg}")
        logger.debug(f"Using sample size {self.sample_size}")
        metric = self.tuner.run_with_error_inj(cfg, self.sample_size)
        diff = abs(metric - self.metric_target)
        logger.debug(f"metric_value={metric:.3f}, distance_to_goal={diff:.3f}")
        result = Result(time=diff)
        self.configs.append((cfg, metric))

        update = self._progress_getter() - self._pbar.n if self._progress_getter else 1
        self._pbar.update(update)
        return result

    def save_final_config(self, config):
        self._pbar.close()
        self.configs = sorted(
            self.configs, key=lambda x: abs(x[1] - self.metric_target)
        )
