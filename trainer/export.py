import numpy as np
import onnx
import onnxruntime as ort
import torch
from omegaconf import DictConfig

from trainer.utils.config import instantiate_key_from_config

# HARDCODING VERSION...
# TODO: 나중에 개선하기


def main(config, weight: str, output: str):
    model = instantiate_key_from_config(config, 'config_model')
    weights = torch.load(weight, map_location='cpu')['state_dict']
    new_weights = {}
    for name, weight in weights.items():
        if 'model' in name:
            name = name.replace('model.', '')
        new_weights[name] = weight
    model.load_state_dict(new_weights)
    model = model.cpu().eval()
    model.header.return_logits = False

    dummy = torch.randn(1, 1, 32, 32)
    torch_output = model(dummy)
    print(torch_output.shape)
    torch.onnx.export(model, dummy, output, verbose=False,
                      input_names=['input'], output_names=['output'], opset_version=11)

    onnx_model = onnx.load(output)
    onnx.checker.check_model(onnx_model)
    onnx_runtime = ort.InferenceSession(output)
    onnx_input = {'input': dummy.numpy()}
    onnx_output = onnx_runtime.run(None, onnx_input)
    print("ONNX output shape:", onnx_output[0].shape)

    # PyTorch와 ONNX 출력 비교
    torch_output_np = torch_output.detach().numpy()
    onnx_output_np = onnx_output[0]
    np.testing.assert_allclose(torch_output_np, onnx_output_np, atol=1e-04)


if __name__ == '__main__':
    import argparse

    from trainer.utils.config import load_config
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, help='config file used for training')
    parser.add_argument('--weight', type=str, required=True)
    parser.add_argument('--output', type=str, required=True)
    args = parser.parse_args()
    config: DictConfig = load_config(args.config)

    main(config, args.weight, args.output)
