import random


def get_random_size(base_size: tuple[int, int], min_scale: float = 0.5, max_scale: float = 2.0) -> tuple[int, int]:
    """Get random size that is multiple of 32 between 0.5x and 2x of base size"""
    h, w = base_size

    # 0.5 ~ 2.0 사이의 random scale factor
    scale = random.uniform(0.5, 2.0)

    # 32의 배수로 반올림
    new_h = int(round((h * scale) / 32) * 32)
    new_w = int(round((w * scale) / 32) * 32)

    return (new_h, new_w)
