import argparse

import timm


def show_encoder_list(keyword: str):
    print(f"Searching for '{keyword}' in the model list...")

    model_list = timm.list_models(pretrained=True)
    model_list = [model for model in model_list if keyword in model]
    print(f"Found {len(model_list)} models:")
    for model in model_list:
        print(model)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=str, default='resnet18')
    args = parser.parse_args()
    show_encoder_list(args.model.lower())
