import argparse

import cv2
import mlflow.pyfunc
import numpy as np

MODEL_NAME = 'test-01'
MODEL_VERSION = 1
MODEL = mlflow.pyfunc.load_model(model_uri=f'models:/{MODEL_NAME}/{MODEL_VERSION}')


def predict_image(image_path):
    image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    image = cv2.resize(image, (32, 32))
    if len(image.shape) == 2:
        image = np.expand_dims(image, axis=2)
    image[image > 0] = 255
    image = image.astype('float32') / 255.0
    image = np.transpose(image, (2, 0, 1))
    image = np.expand_dims(image, axis=0)

    result = MODEL.predict(image)['output']

    # print
    image = np.squeeze(image, axis=(0, 1))
    image = cv2.resize(image, (20, 20))
    for r in range(20):
        for c in range(20):
            if image[r][c] != 0:
                print("ㅁ", end=' ')
            else:
                print("  ", end=' ')
        print()

    predicted_value = np.argmax(result)
    print(f"\nPredicted class: {predicted_value}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Predict image class using ONNX model")
    parser.add_argument("--image-path", type=str, required=True, help="Path to the input image file")
    args = parser.parse_args()

    predict_image(args.image_path)
