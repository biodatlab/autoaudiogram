from pycaret.classification import *
import pandas as pd
import argparse
import os

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--path", type=str, default="data/data_extract.csv", help="path to data file"
    )
    parser.add_argument(
        "--target", type=str, default="label", help="target column name"
    )
    parser.add_argument("--gpu", type=bool, default=True, help="use gpu")
    parser.add_argument(
        "--experiment_name", type=str, default="extract", help="experiment name"
    )
    args = parser.parse_args()

    # Load data
    data_path = args.path
    target = args.target
    use_gpu = args.gpu
    experiment_name = args.experiment_name
    data = pd.read_csv(data_path)

    # setup
    clf = setup(
        data=data,
        target=target,
        use_gpu=use_gpu,
        session_id=123,
        experiment_name=experiment_name,
    )
    best = compare_models()

    # save model
    os.makedirs("models", exist_ok=True)
    save_path = os.path.join("models", experiment_name + "_model")
    save_model(best, save_path)

    # evaluate model
    eval_dir = os.path.join("evaluate", experiment_name)
    os.makedirs(eval_dir, exist_ok=True)
    cf = plot_model(best, plot="confusion_matrix", save=True)
    roc = plot_model(best, save=True)
    feature_importance = plot_model(best, plot="feature", save=True)

    # move evaluate files to evaluate folder
    os.rename("Confusion Matrix.png", os.path.join(eval_dir, "confusion_matrix.png"))
    os.rename("AUC.png", os.path.join(eval_dir, "roc.png"))
    os.rename(
        "Feature Importance.png", os.path.join(eval_dir, "feature_importance.png")
    )
