
import pickle
import timeit
from typing import Optional

import numpy as np
import onnxruntime as ort

from .base import BenchmarkResults, DeviceType, MetaData
from .excutor import (ArrayLike, BaseExecutor, Device, DeviceError,
                      ModelLoadError, PathLike)


class ONNXExecutor(BaseExecutor[ort.InferenceSession]):
    """
    ONNX-specific implementation of Neural Network Converter.
    Converts and runs ONNX models with optimized performance.

    Additional Args:
        io_binding: Enable IO binding for GPU optimization
    """

    def __init__(self, model_path: PathLike, device: Device = DeviceType.CPU.value,
                 meta_path: Optional[PathLike] = None, io_binding: bool = False):
        self.io_binding = self._validate_io_binding(device, io_binding)
        super().__init__(model_path, device, meta_path)

    def _validate_io_binding(self, device: Device, io_binding: bool) -> bool:
        if io_binding and device == DeviceType.CPU.value:
            raise DeviceError("IO-binding is only supported for GPU devices")
        return io_binding

    def _get_providers(self) -> list[str]:
        return (['CPUExecutionProvider']if self.device == DeviceType.CPU.value else ['CUDAExecutionProvider'])

    def _load_model(self) -> None:
        """Load ONNX model."""
        try:
            sess_options = ort.SessionOptions()
            sess_options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL

            self.model = ort.InferenceSession(str(self.model_path), sess_options,
                                              providers=self._get_providers())

            # Store input/output info
            self._input_names = [input.name for input in self.model.get_inputs()]
            self._output_names = [output.name for output in self.model.get_outputs()]

            # Validate against metadata if available
            if self.meta:
                if self.meta.input_names and set(self.meta.input_names) != set(self._input_names):
                    print(f"Warning: Metadata input names {self.meta.input_names} "
                          f"don't match model inputs {self._input_names}")

                if self.meta.output_names and set(self.meta.output_names) != set(self._output_names):
                    print(f"Warning: Metadata output names {self.meta.output_names} "
                          f"don't match model outputs {self._output_names}")

        except Exception as e:
            raise ModelLoadError(f"Failed to load ONNX model: {str(e)}") from e

    def _run(self, inputs: ArrayLike):
        """Execute ONNX model inference."""
        if not isinstance(inputs, (np.ndarray, list)):
            raise TypeError(f"Invalid input type: {type(inputs)}")

        input_list = [inputs] if isinstance(inputs, np.ndarray) else inputs
        if len(input_list) != len(self.model.get_inputs()):
            raise ValueError(f"Input count mismatch: expected {len(self.model.get_inputs())}, "
                             f"got {len(input_list)}")

        try:
            if self.io_binding:
                outputs = self._run_with_io_binding(input_list)
            else:
                outputs = self._run_without_io_binding(input_list)
        except Exception as e:
            raise RuntimeError(f"Inference failed: {str(e)}") from e
        return outputs

    def _run_without_io_binding(self, inputs: list[np.ndarray]) -> dict[str, np.ndarray]:
        """Execute inference without IO binding."""
        ort_inputs = dict(zip(self._input_names, inputs))
        outputs = self.model.run(None, ort_inputs)
        return dict(zip(self._output_names, outputs))

    def _run_with_io_binding(self, inputs: list[np.ndarray]) -> dict[str, np.ndarray]:
        """Execute inference with IO binding for optimized GPU performance."""
        raise NotImplementedError("IO binding not yet implemented")

    def benchmark(self, sample_input: Optional[ArrayLike] = None,
                  num_iterations: int = 50, num_warmup: int = 3) -> BenchmarkResults:
        """
        Benchmark model performance.

        Args:
            sample_input: Sample input for benchmarking (if None, random input is used)
            num_iterations: Number of benchmark iterations
            num_warmup: Number of warmup iterations
        """
        if sample_input is None:
            inputs = [np.random.randn(*input.shape).astype(np.float32) for input in self.model.get_inputs()]
        else:
            inputs = sample_input if isinstance(sample_input, list) else [sample_input]

        # Warm up
        for _ in range(num_warmup):
            self.run(inputs)

        # Benchmark
        latencies = []
        for _ in range(num_iterations):
            start_time = timeit.default_timer()
            _ = self.run(inputs)
            latencies.append(timeit.default_timer() - start_time)

        avg_latency = sum(latencies) / len(latencies)
        return BenchmarkResults(fps=1000.0 / avg_latency, avg=avg_latency,
                                min=min(latencies), max=max(latencies))

    @property
    def input_names(self) -> list[str]:
        """Get model input names."""
        return self._input_names

    @property
    def output_names(self) -> list[str]:
        """Get model output names."""
        return self._output_names
