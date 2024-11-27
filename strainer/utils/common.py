from typing import Any

import torch


def unwraps_keys(dictionary: dict[str, Any], src_key: str, dst_key: str) -> dict:
    """
    Change all dictionary keys containing a specific keyword to another keyword.
    """
    updated_dict = {}
    for key, value in dictionary.items():
        updated_key = key.replace(src_key, dst_key)
        updated_dict[updated_key] = value
    return updated_dict


def load_weights(path: str, src_key: str = '', dst_key: str = '') -> dict:
    weight = torch.load(path, map_location='cpu', weights_only=False)
    weight = weight['state_dict'] if 'state_dict' in weight else weight
    weight = unwraps_keys(weight, src_key, dst_key)
    return weight


def get_device(device: str | int) -> torch.device:
    if device == 'cpu':
        return torch.device('cpu')
    if not torch.cuda.is_available():
        raise ValueError('Cuda is not available')
    if device == 'cuda':
        return torch.device('cuda:0')
    elif isinstance(device, int) or device.isdigit():
        return torch.device(f'cuda:{device}')
    else:
        raise ValueError('Invalid device')
