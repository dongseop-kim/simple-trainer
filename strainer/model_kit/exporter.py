"""
Neural network model export utilities.
Provides functionality to export PyTorch models to various formats.
"""
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Optional, Union

import torch
import torch.nn as nn
from torch import Tensor

from .base import Framework, MetaData, PathLike
from .onnx import ONNXExecutor


@dataclass
class ExportOutput:
    """Container for export file paths."""
    model_path: PathLike
    meta_path: PathLike

    def __post_init__(self):
        """Convert paths to Path objects."""
        self.model_path = Path(self.model_path)
        self.meta_path = Path(self.meta_path)


class ExportError(Exception):
    """Base exception for export-related errors."""
    pass


class Exporter:
    """
    Exports PyTorch models to various formats.

    Args:
        checkpoint_path: Path to model checkpoint
        model: PyTorch model class or instance

    Example:
        ```python
        exporter = Exporter('model.pth', MyModel())
        output = exporter.export_onnx(
            save_dir='exported_models',
            model_name='my_model',
            dummy_inputs=torch.randn(1, 3, 224, 224)
        )
        print(f"Model saved to: {output.model_path}")
        print(f"Metadata saved to: {output.meta_path}")
        ```
    """

    OPSET_VERSION = 17  # ONNX opset version

    def __init__(self, model: nn.Module, checkpoint_path: Optional[PathLike] = None):
        self.model = model

        if checkpoint_path:
            self.checkpoint_path = Path(checkpoint_path)
            self._load_checkpoint()

        self.model = self.model.eval().cpu()

    def _validate_path(self) -> None:
        """Validate checkpoint file exists."""
        if not self.checkpoint_path.exists():
            raise FileNotFoundError(f"Checkpoint not found: {self.checkpoint_path}")

    def _load_checkpoint(self) -> None:
        """Load model from checkpoint."""
        try:
            # Load checkpoint
            ckpt = torch.load(self.checkpoint_path, map_location='cpu')
            state_dict = ckpt['state_dict'] if 'state_dict' in ckpt else ckpt

            # Get model instance
            model = self.model_def.model if hasattr(self.model_def, 'model') else self.model_def

            # Load weights and prepare model
            model.load_state_dict(state_dict, strict=True)
            self.model = model.eval().cpu()

        except Exception as e:
            raise ExportError(f"Failed to load checkpoint: {str(e)}") from e

    def _create_save_paths(self, save_dir: PathLike, model_name: str, framework: Framework) -> tuple[Path, Path]:
        """Create and return paths for model and metadata files."""
        save_dir = Path(save_dir) / framework.value
        save_dir.mkdir(parents=True, exist_ok=True)

        return (save_dir / f"{model_name}.{framework.value}", save_dir / f"{model_name}_meta.json")

    def _prepare_io_names(self, dummy_inputs: Tensor | list[Tensor]) -> tuple[list[str], list[str]]:
        """
        Prepare input and output names based on model structure.

        Args:
            dummy_inputs: Sample inputs for tracing

        Returns:
            Tuple of (input_names, output_names)
        """
        # Handle input names
        input_count = len(dummy_inputs) if isinstance(dummy_inputs, list) else 1
        input_names = [f'input_{i}' for i in range(input_count)]

        # Get output count by tracing the model
        try:
            with torch.no_grad():
                outputs = self.model(dummy_inputs)
                output_count = len(outputs) if isinstance(outputs, (list, tuple)) else 1
                output_names = [f'output_{i}' for i in range(output_count)]

        except Exception as e:
            raise ExportError(f"Failed to trace model outputs: {str(e)}") from e
        print()
        return input_names, output_names

    def export_onnx(self, save_dir: PathLike, model_name: str, dummy_inputs: Tensor | list[Tensor]) -> ExportOutput:
        """
        Export model to ONNX format with metadata file.

        Args:
            save_dir: Directory to save exported files
            model_name: Name for the exported model
            dummy_inputs: Sample inputs for tracing

        Returns:
            ExportOutput containing paths to saved files

        Raises:
            ExportError: If export process fails at any stage
        """
        # Prepare paths and metadata
        model_path, meta_path = self._create_save_paths(save_dir, model_name, Framework.ONNX)
        input_names, output_names = self._prepare_io_names(dummy_inputs)

        meta = MetaData(framework=Framework.ONNX.value, model_name=model_name,
                        input_names=input_names, output_names=output_names,
                        version={'opset_version': self.OPSET_VERSION})

        # Export ONNX model
        try:
            torch.onnx.export(model=self.model, args=dummy_inputs, f=model_path,
                              input_names=input_names, output_names=output_names,
                              opset_version=self.OPSET_VERSION,
                              do_constant_folding=True,
                              )
            # save meta
            with open(meta_path, 'w') as f:
                json.dump(meta.to_dict(), f, indent=2)
        except Exception as e:
            raise ExportError(f"Failed to export ONNX model: {str(e)}") from e

        # Collect model info and benchmark
        try:
            onnx_executor = ONNXExecutor(model_path, device='cpu', meta_path=meta_path)
            results: dict[str, float] = onnx_executor.benchmark()

            # Update metadata with model info
            meta.input_shapes = onnx_executor.meta.input_shapes
            meta.output_shapes = onnx_executor.meta.output_shapes
            meta.benchmark = results

        except Exception as e:
            raise ExportError(f"Failed to collect model information: {str(e)}") from e

        # Save metadata
        try:
            with open(meta_path, 'w') as f:
                json.dump(meta.to_dict(), f, indent=2)
        except Exception as e:
            raise ExportError(f"Failed to save metadata: {str(e)}") from e

        return ExportOutput(model_path=model_path, meta_path=meta_path)

    @ staticmethod
    def load_metadata(meta_path: PathLike) -> dict[str, Any]:
        """
        Load metadata from JSON file.

        Args:
            meta_path: Path to metadata JSON file

        Returns:
            Dictionary containing metadata

        Raises:
            ExportError: If metadata loading fails
        """
        try:
            with open(meta_path, 'r') as f:
                return json.load(f)
        except Exception as e:
            raise ExportError(f"Failed to load metadata: {str(e)}") from e
