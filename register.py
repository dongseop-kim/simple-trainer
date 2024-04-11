import argparse

import mlflow
import mlflow.onnx
import onnx


def register_model(experiment_id, model_name, model_path):
    mlflow.set_tracking_uri('./mlruns')
    experiment = mlflow.get_experiment_by_name(experiment_id)

    if experiment:
        experiment_id = experiment.experiment_id
    else:
        experiment_id = mlflow.create_experiment(experiment_id)

    with mlflow.start_run(experiment_id=experiment_id) as run:
        model = onnx.load_model(model_path)
        mlflow.onnx.log_model(model, model_name)

    print(f"Model '{model_name}' registered in experiment '{experiment_id}'")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Register a model in MLflow")
    parser.add_argument("--experiment-id", type=str, required=True, help="The ID of the experiment")
    parser.add_argument("--model-name", type=str, required=True, help="The name for the registered model")
    parser.add_argument("--model-path", type=str, required=True, help="The path to the model")

    args = parser.parse_args()

    register_model(args.experiment_id, args.model_name, args.model_path)
