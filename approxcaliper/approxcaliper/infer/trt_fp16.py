"""Customized version of TensorRTBackendRep taken from onnx_tensorrt"""

import numpy as np
import onnx
import six
import tensorrt as trt
from onnx.backend.base import BackendRep, Device, DeviceType, namedtupledict
from .backend import TRT_LOGGER, cudaSetDevice
from .tensorrt_engine import Engine


class TensorRTBackendRep(BackendRep):
    def __init__(
        self,
        model,
        device,
        max_batch_size=32,
        max_workspace_size=None,
        dtype=np.float32,
        serial_trt=None,
    ):
        if not isinstance(device, Device):
            device = Device(device)
        self._set_device(device)
        self._logger = TRT_LOGGER
        self.builder = trt.Builder(self._logger)
        self.network = self.builder.create_network(
            flags=1 << (int)(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH)
        )
        self.parser = trt.OnnxParser(self.network, self._logger)

        if not isinstance(model, six.string_types):
            model_str = model.SerializeToString()
        else:
            model_str = model

        if not trt.init_libnvinfer_plugins(TRT_LOGGER, ""):
            msg = "Failed to initialize TensorRT's plugin library."
            raise RuntimeError(msg)

        if not self.parser.parse(model_str):
            error = self.parser.get_error(0)
            msg = "While parsing node number %i:\n" % error.node()
            msg += "%s:%i In function %s:\n[%i] %s" % (
                error.file(),
                error.line(),
                error.func(),
                error.code(),
                error.desc(),
            )
            raise RuntimeError(msg)

        self.builder.max_batch_size = max_batch_size
        config = self.builder.create_builder_config()
        if max_workspace_size is None:
            max_workspace_size = 1 << 28
        config.max_workspace_size = max_workspace_size
        if dtype == np.float32:
            pass
        elif dtype == np.float16:
            config.flags = config.flags | int(trt.BuilderFlag.FP16) | int(trt.BuilderFlag.STRICT_TYPES)
        else:
            raise ValueError(f"dtype {dtype} not understood")

        if not serial_trt:
            self.runtime = None
            trt_engine = self.builder.build_engine(self.network, config)
            if trt_engine is None:
                raise RuntimeError("Failed to build TensorRT engine from network")
        else:
            self.runtime = trt.Runtime(TRT_LOGGER)
            trt_engine = self.runtime.deserialize_cuda_engine(serial_trt)
        self.engine = Engine(trt_engine)

        self._output_shapes = {}
        self._output_dtype = {}
        for output in model.graph.output:
            dims = output.type.tensor_type.shape.dim
            output_shape = tuple([dim.dim_value for dim in dims])
            self._output_shapes[output.name] = output_shape
            self._output_dtype[output.name] = output.type.tensor_type.elem_type

    def _set_device(self, device):
        self.device = device
        assert device.type == DeviceType.CUDA
        cudaSetDevice(device.device_id)

    def run(self, inputs, **kwargs):
        """Execute the prepared engine and return the outputs as a named tuple.
        inputs -- Input tensor(s) as a Numpy array or list of Numpy arrays.
        """
        if isinstance(inputs, np.ndarray):
            inputs = [inputs]
        outputs = self.engine.run(inputs)
        output_names = [output.name for output in self.engine.outputs]

        for i, (name, array) in enumerate(zip(output_names, outputs)):
            output_shape = self._output_shapes[name]
            # HACK WAR replace fixed batch dim with variable
            if (
                self._output_dtype[name] == onnx.TensorProto.INT64
                and array.dtype == np.int32
            ):
                casted_output = np.array(outputs[i], dtype=np.int64)
                if np.equal(outputs[i], casted_output).all():
                    outputs[i] = np.array(outputs[i], dtype=np.int64)
        return namedtupledict("Outputs", output_names)(*outputs)
