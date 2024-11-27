import json
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any, Generic, Optional, TypeVar, Union

import numpy as np

from .base import BenchmarkResults, DeviceType, Framework, MetaData, PathLike

# Type definitions
T = TypeVar('T')
Device = Union[str, int]
ArrayLike = Union[np.ndarray, list[np.ndarray]]


class ModelLoadError(Exception):
    pass


class DeviceError(Exception):
    """Raised when device configuration is invalid."""
    pass


class BaseExecutor(ABC, Generic[T]):
    """
    Base Neural Network Converter class providing common functionality for model operations.
    Converts and runs different types of neural network models with consistent interface.

    Args:
        fname: Path to the model file
        device: Target device for model execution ('cpu', 'gpu', or GPU device number)
    """

    AVAILABLE_FRAMEWORKS = [f.value for f in Framework]

    def __init__(self, model_path: PathLike,
                 device: Device = DeviceType.CPU.value,
                 meta_path: Optional[PathLike] = None):
        self.model_path = Path(model_path)
        self.device = self._validate_device(device)
        self.meta_path = Path(meta_path) if meta_path else None
        self.meta: Optional[MetaData] = None
        self.model: Optional[T] = None

        self._validate_paths()
        self._load_metadata()
        self.load_model()

    def _validate_device(self, device: Device) -> str:
        """Validate and normalize device specification."""
        if isinstance(device, str):
            device_lower = device.lower()
            if device_lower not in [DeviceType.CPU.value, DeviceType.GPU.value]:
                raise DeviceError(f"Invalid device string: {device}. "
                                  f"Must be '{DeviceType.CPU.value}' or '{DeviceType.GPU.value}'")
            return device_lower

        if isinstance(device, int):
            if device < 0:
                raise DeviceError(f"Invalid device number: {device}. Must be non-negative.")
            return str(device)

    def _validate_paths(self) -> None:
        """Validate file paths."""
        if not self.model_path.exists():
            raise FileNotFoundError(f"Model file not found: {self.model_path}")

        if self.meta_path and not self.meta_path.exists():
            raise FileNotFoundError(f"Metadata file not found: {self.meta_path}")

    def _load_metadata(self) -> None:
        """Load metadata if available."""
        if self.meta_path:
            try:
                with open(self.meta_path, 'r') as f:
                    self.meta = MetaData(**json.load(f))
            except Exception as e:
                print(f"Warning: Failed to load metadata: {str(e)}")
                self.meta = None

    def load_model(self) -> None:
        try:
            self._load_model()
        except Exception as e:
            raise ModelLoadError(f"Failed to load model: {str(e)}") from e

    def run(self, inputs: Any) -> Any:
        if self.model is None:
            raise RuntimeError("Model not loaded. Call load_model() first.")
        return self._run(inputs)

    def __call__(self, inputs: Any) -> Any:
        return self.run(inputs)

    @abstractmethod
    def _load_model(self):
        pass

    @abstractmethod
    def _run(self, inputs: ArrayLike):
        pass

    @abstractmethod
    def benchmark(self, sample_input: Optional[ArrayLike] = None,
                  num_iterations: int = 50, num_warmup: int = 3) -> BenchmarkResults:
        """Benchmark model performance."""
        pass
