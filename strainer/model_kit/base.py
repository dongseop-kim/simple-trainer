"""
Metadata and environment information utilities for model export and execution.
Provides functionality to collect system information and manage model metadata.
"""

import platform
import subprocess
import sys
from collections import OrderedDict
from dataclasses import dataclass, field
from enum import Enum, auto
from pathlib import Path
from typing import Any, Dict, List, Optional

PathLike = str | Path


class Framework(str, Enum):
    """Supported deep learning frameworks."""
    ONNX = 'onnx'
    # OPENVINO = 'openvino'  # to be implemented
    # TORCHSCRIPT = 'torchscript'  # to be implemented


class DeviceType(str, Enum):
    """Supported device types for model execution."""
    CPU = 'cpu'
    GPU = 'gpu'
    CUDA = auto()  # For device numbers


class SystemInfo:
    """Utility class to collect system information."""

    @staticmethod
    def get_git_commit() -> str:
        """Get current git commit hash."""
        try:
            result = subprocess.run(['git', 'rev-parse', 'HEAD'],
                                    capture_output=True, text=True, check=True)
            return result.stdout.strip()
        except (subprocess.SubprocessError, FileNotFoundError):
            return 'Unknown'

    @staticmethod
    def get_python_version() -> str:
        """Get current Python version."""
        return f'{sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro}'

    @staticmethod
    def get_cpu_info() -> str:
        """Get CPU model name."""
        if platform.system() != 'Linux':
            return 'Unknown'

        try:
            result = subprocess.run(['lscpu'], capture_output=True, text=True, check=True)

            for line in result.stdout.splitlines():
                if line.startswith('Model name:'):
                    return line.split(':', 1)[1].strip()
            return 'Unknown'

        except subprocess.SubprocessError:
            return 'Unknown'

    @staticmethod
    def get_gpu_info() -> str:
        """Get GPU model name."""
        try:
            import GPUtil
            gpus = GPUtil.getGPUs()
            return gpus[0].name if gpus else 'Unknown'
        except (ImportError, Exception):
            return 'Unknown'

    @staticmethod
    def get_os_version() -> str:
        """Get Ubuntu version."""
        if platform.system() != 'Linux':
            return 'Unknown'

        try:
            with open('/etc/os-release', 'r') as f:
                for line in f:
                    if line.startswith('VERSION_ID='):
                        return line.strip().split('=')[1].strip('"')
            return 'Unknown'
        except FileNotFoundError:
            return 'Unknown'


@dataclass
class BenchmarkResults:
    """Container for model benchmark results."""
    fps: Optional[float] = None
    avg: Optional[float] = None
    min: Optional[float] = None
    max: Optional[float] = None

    def to_dict(self) -> Dict[str, Optional[float]]:
        return {'fps': self.fps, 'avg': self.avg, 'min': self.min, 'max': self.max}


@dataclass
class MetaData:
    """
    Container for model metadata and system information.

    Attributes:
        framework: Deep learning framework used
        model_name: Name of the model
        model: Serialized model bytes
        input_names: List of input tensor names
        output_names: List of output tensor names
        input_shapes: Dictionary of input shapes
        output_shapes: Dictionary of output shapes
    """

    framework: Framework
    model_name: Optional[str] = None
    model: Optional[bytes] = None
    input_names: Optional[List[str]] = None
    output_names: Optional[List[str]] = None
    input_shapes: Optional[Dict[str, str]] = None
    output_shapes: Optional[Dict[str, str]] = None

    version: Dict[str, str] = field(default_factory=lambda: {'os': SystemInfo.get_os_version(),
                                                             'python': SystemInfo.get_python_version()})

    devices: Dict[str, str] = field(default_factory=lambda: {'cpu': SystemInfo.get_cpu_info(),
                                                             'gpu': SystemInfo.get_gpu_info()})

    commit_hash: str = field(default_factory=SystemInfo.get_git_commit)
    benchmark: BenchmarkResults = field(default_factory=BenchmarkResults)

    def to_dict(self, include_model: bool = True) -> Dict[str, Any]:
        """
        Convert metadata to dictionary.

        Args:
            include_model: Whether to include model bytes in output

        Returns:
            Dictionary representation of metadata
        """
        return OrderedDict({'framework': self.framework,
                            'version': self.version,
                            'devices': self.devices,
                            'commit_hash': self.commit_hash,
                            'model_name': self.model_name,
                            'model': self.model if include_model else None,
                            'input_names': self.input_names,
                            'input_shapes': self.input_shapes,
                            'output_names': self.output_names,
                            'output_shapes': self.output_shapes,
                            'benchmark': self.benchmark.to_dict()})

    def update_benchmark(self, results: BenchmarkResults) -> None:
        """Update benchmark results."""
        self.benchmark = results

    def update_model_info(self,
                          input_names: List[str], output_names: List[str],
                          input_shapes: Dict[str, str], output_shapes: Dict[str, str]) -> None:
        """Update model information."""
        self.input_names = input_names
        self.output_names = output_names
        self.input_shapes = input_shapes
        self.output_shapes = output_shapes
