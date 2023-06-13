import logging
import os
import time
from pathlib import Path
from typing import Dict, Iterable, Union
from collections import OrderedDict

import numpy as np

PathLike = Union[str, Path]
logger = logging.getLogger(__name__)


def onnx_to_openvino(
    onnx_file: Path, openvino_cmd: str, input_shape: Iterable[int], output_path: Path
) -> Path:
    import subprocess

    input_shape_str = ",".join(str(x) for x in input_shape)
    input_shape_str = f"[{input_shape_str}]"
    os.makedirs(output_path)
    subprocess.run(
        [
            openvino_cmd,
            "--input_model",
            onnx_file.as_posix(),
            "--input_shape",
            input_shape_str,
            "--output_dir",
            output_path.as_posix(),
        ],
        check=True,
    )  # Raise if failed
    return output_path / onnx_file.with_suffix(".xml").name


def find_or_get_openvino_path(
    filepath: Path, openvino_cmd: str, input_shape: Iterable[int]
) -> Path:
    if filepath.is_file() and filepath.suffix == ".onnx":
        output_dir = filepath.parent / f"{filepath.stem}_openvino"
        openvino_xml = output_dir / filepath.with_suffix(".xml").name
        if not openvino_xml.is_file():
            onnx_to_openvino(filepath, openvino_cmd, input_shape, output_dir)
        return openvino_xml
    elif filepath.is_dir():
        xml_files = list(filepath.glob("*.xml"))
        if len(xml_files) == 1:
            (xml_file,) = xml_files
            if xml_file.with_suffix(".bin").is_file():
                return xml_file
    raise ValueError(f"Cannot find openvino files in {filepath}")


class Timer:
    def __init__(self):
        self.start_time = None
        self.tdelta = None

    def __enter__(self):
        self.start_time = time.time()
        return self

    def __exit__(self, n1, n2, n3):
        self.tdelta = time.time() - self.start_time
        self.start_time = None


def print_time(format_str, print_method=print):
    def decorator(timed_func):
        def timing_func(*args, **kwargs):
            timer = Timer()
            with timer:
                ret = timed_func(*args, **kwargs)
            print_method(format_str.format(timer.tdelta))
            return ret

        return timing_func

    return decorator


class CallableModel:
    iecore_inst = None

    def __init__(self, forward_fn):
        self.forward = forward_fn

    def __call__(self, *args, **kwargs):
        return self.forward(*args, **kwargs)

    @classmethod
    def from_backend(cls, backend: str, model_path: PathLike, **kwargs):
        backend = backend.lower()
        if backend == "onnx":
            return cls.from_onnx(model_path, **kwargs)
        elif backend == "openvino":
            return cls.from_openvino(model_path, **kwargs)
        elif backend == "tensorrt":
            return cls.from_tensorrt(model_path, **kwargs)
        else:
            raise ValueError(f"Unknown backend: {backend}")

    @classmethod
    def from_onnx(
        cls,
        onnx_file_path: PathLike,
        device: str,
        output_name_map: Dict[str, str] = None,
    ) -> "CallableModel":
        import onnxruntime as oxr

        PROVIDER_NAME = {
            "GPU": "CUDAExecutionProvider",
            "CPU": "CPUExecutionProvider",
            "TensorRT": "TensorRTExecutionProvider",
        }
        all_providers = set(oxr.get_available_providers())
        provider = PROVIDER_NAME[device]
        if provider not in all_providers:
            raise ValueError(
                f"{device} ({provider}) is not available. "
                f"Available providers: {all_providers}"
            )
        logger.info(f"Using device {device} ({provider}) for onnx inference")
        session = oxr.InferenceSession(
            Path(onnx_file_path).as_posix(), providers=[provider]
        )

        inputs = [input_.name for input_ in session.get_inputs()]
        if len(inputs) != 1:
            raise ValueError(f"Only 1 input is supported, got {len(inputs)}")
        input_name = inputs[0]
        output_name_map = output_name_map or {}
        output_names = OrderedDict(
            (output.name, output_name_map.get(output.name, output.name))
            for output in session.get_outputs()
        )
        logger.info(
            f"ONNX model {onnx_file_path}: {inputs[0]} -> {cls._pformat_output_names(output_names)}"
        )

        def run_on_data(data: np.ndarray):
            io_binding = session.io_binding()
            data = data.astype(np.float32)
            io_binding.bind_cpu_input(input_name, data)
            for output_name in output_names:
                io_binding.bind_output(output_name)
            session.run_with_iobinding(io_binding)
            output_values = io_binding.copy_outputs_to_cpu()
            return dict(zip(output_names.values(), output_values))

        return cls(run_on_data)

    @classmethod
    def _pformat_output_names(cls, outputs: Dict[str, str]):
        return str(
            [
                from_name if from_name == to_name else f"{from_name} ({to_name})"
                for from_name, to_name in outputs.items()
            ]
        )

    @classmethod
    def from_openvino(
        cls, onnx_or_openvino: PathLike, device: str, openvino_cmd: str
    ) -> "CallableModel":
        from openvino.inference_engine import IECore

        onnx_or_openvino = Path(onnx_or_openvino)
        openvino_xml = find_or_get_openvino_path(onnx_or_openvino, openvino_cmd, 1)
        if cls.iecore_inst is None:
            cls.iecore_inst = IECore()
        net = cls.iecore_inst.read_network(
            model=openvino_xml.as_posix(),
            weights=openvino_xml.with_suffix(".bin").as_posix(),
        )
        exec_net = cls.iecore_inst.load_network(network=net, device_name=device)
        input_blob = next(iter(exec_net.inputs))
        out_blob = next(iter(exec_net.outputs))
        # OpenVINO returns a dictionary from output name to output
        # We flatten it to a list of return values (discarding keys)
        # in coherence with onnx return values
        return cls(
            lambda data: list(exec_net.infer(inputs={input_blob: data}).values())
        )

    @classmethod
    @print_time("Tensorrt init took {:.2f} seconds")
    def from_tensorrt(cls, model_file: PathLike) -> "CallableModel":
        import pickle
        from .trt_fp16 import TensorRTBackendRep

        model_file = Path(model_file)
        # If given file is not an ONNX file, use it like a compiled TRT model file (serialized engine)
        if model_file.suffix != ".onnx":
            compiled_pkl_file = model_file
        # Otherwise first see if there's a .pkl in the same directory with the same name
        else:
            compiled_pkl_file = model_file.with_suffix(".pkl")
            # If that doesn't exist, create it (batch size is assumed to be 1)
            if not compiled_pkl_file.is_file():
                print(f"Compiling {model_file} to {compiled_pkl_file}")
                compile_onnx_to_trt(1, model_file, compiled_pkl_file)
        with compiled_pkl_file.open("rb") as f:
            obj = pickle.load(f)
        print(f"Loading from compiled model {compiled_pkl_file}")
        backend = TensorRTBackendRep(
            obj["model"],
            device="CUDA:0",
            dtype=np.float16,
            serial_trt=obj["engine"],
        )
        return cls(
            lambda data: backend.run(np.ascontiguousarray(data, dtype=np.float32))
        )

    @classmethod
    def from_pytorch(cls, model, checkpoint: PathLike, device: str) -> "CallableModel":
        import torch

        checkpoint_dict = torch.load(checkpoint, map_location=device)
        model.load_state_dict(checkpoint_dict.pop("net"), strict=False)
        model = model.to(device)

        def run_model(np_input: np.ndarray):
            input_ = torch.tensor(np_input, device=device).float()
            return model(input_).detach().cpu().numpy()

        return cls(run_model)


@print_time("Model compilation took {:.2f} seconds")
def compile_onnx_to_trt(batchsize: int, infile: Path, outfile: Path):
    import pickle
    import numpy as np
    import onnx
    from .trt_fp16 import TensorRTBackendRep

    model = onnx.load(infile)
    # The following code changes the first dimension of every input to be batch-dim
    # Modify as appropriate ... note that this requires all inputs to
    # have the same batch_dim
    inputs = model.graph.input
    for input_ in inputs:
        # Checks omitted.This assumes that all inputs are tensors and have a shape with first dim.
        # Add checks as needed.
        dim1 = input_.type.tensor_type.shape.dim[0]
        dim1.dim_value = batchsize
    trt_engine = TensorRTBackendRep(
        model, device="CUDA:0", dtype=np.float16
    ).engine.engine
    with outfile.open("wb") as f:
        pickle.dump({"model": model, "engine": bytearray(trt_engine.serialize())}, f)
