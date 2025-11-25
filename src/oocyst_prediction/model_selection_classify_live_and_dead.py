#!/usr/bin/env python3
"""
This script processes live and dead oocyst segmentations by extracting features using the SimpleITK
LabelIntensityStatisticsImageFilter, training multiple machine learning models including Random Forest,
SVM, and Gradient Boosting, and evaluating their performance using AUC, Precision, and Recall metrics
for both live and dead oocysts. The best model is then selected based on cross-validation performance
and aggregated statistics across multiple runs.


Usage:
    python model_selection_classify_live_and_dead.py live_and_dead_segmentations.csv results
"""
import sys
import pathlib
import numpy as np
import pandas as pd
import argparse
import tempfile
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.svm import SVC
from sklearn.model_selection import KFold
from sklearn.metrics import roc_auc_score, classification_report
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from skl2onnx.common.data_types import FloatTensorType
from skl2onnx import convert_sklearn
from typing import Tuple, Dict
import src.utils as utils
import shutil


def train_and_evaluate_oocyst_classification_models(
    models: Dict[str, object],
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_test: np.ndarray,
    y_test: np.ndarray,
    model_filename: str,
) -> Dict:
    """
    Train and evaluate multiple oocyst classification models (Random Forest, SVM, Gradient Boosting)
    based on the features (area, perimeter, elongation, flatness , etc.) extracted from live and
    dead oocyst segmentations using SimpleITK LabelIntensityStatisticsImage.


    Args:
        models: Dict of model name to model instance (e.g {"Random Forest": RandomForestClassifier(...), ...})
        X_train: Training data: Feature vectors for (area, perimeter, elongation, flatness, etc.)
        y_train: Labels (live and dead) for each of the feature vectors for training
        X_test: Test data: Feature vectors for (area, perimeter, elongation, flatness, etc.)
        y_test: Labels (live and dead) for each of the feature vectors for testing
        model_filename: Path to save the trained model in .onnx format

    Returns:
        Dict containing model results for each of the models including AUC, optimal threshold,
                                precision, recall, and f1-score
    """

    results = {}

    for name, model in models.items():

        model.fit(X_train, y_train)

        # predict_proba works for all models including SVM (since we set probability=True,and the scores
        # are calibrated using Platt scaling https://scikit-learn.org/stable/modules/svm.html#scores-and-probabilities)
        y_pred_proba = model.predict_proba(X_test)[:, 1]

        auc = roc_auc_score(y_test, y_pred_proba)
        optimal_threshold = utils.calculate_youden_index(y_test, y_pred_proba)

        y_pred = (y_pred_proba >= optimal_threshold).astype(int)

        report = classification_report(
            y_test,
            y_pred,
            output_dict=True,
            labels=[1, 0],
            target_names=["dead", "live"],
        )

        prefix = name.replace(" ", "_").lower()
        onnx_path = (
            pathlib.Path(model_filename).parent
            / f"{prefix}_{pathlib.Path(model_filename).name}"
        )

        initial_type = [("float_input", FloatTensorType([None, X_train.shape[1]]))]

        onnx_model = convert_sklearn(model, initial_types=initial_type)

        onnx_model.metadata_props.add(
            key="optimal_threshold", value=str(optimal_threshold)
        )

        with open(onnx_path, "wb") as f:
            f.write(onnx_model.SerializeToString())

        results[name] = {
            "auc": auc,
            "optimal_threshold": optimal_threshold,
            "live_precision": report["live"]["precision"],
            "live_recall": report["live"]["recall"],
            "live_f1": report["live"]["f1-score"],
            "dead_precision": report["dead"]["precision"],
            "dead_recall": report["dead"]["recall"],
            "dead_f1": report["dead"]["f1-score"],
        }

    return results


def get_all_fold_results(models, df, output_dir, run):
    """
    Perform K-fold cross-validation for each of the runs for all the models and collect results
    for each fold for all the models.
    Args:
        models(dict): Dict of model name to model instance (e.g {"Random Forest": RandomForestClassifier(...), ...})
        df(pd.DataFrame): Dataframe containing all features ('area','elongation', etc.) and labels ('live' and 'dead')
        output_dir(str): Directory to save trained model parameters in .onnx format
        run(int):  Experiment run number
    Returns:
        kfold_results (Dict): Dictionary containing results for each model across folds for a given run

    """
    features = [
        "area",
        "perimeter",
        "elongation",
        "flatness",
        "mean_intensity",
        "min_intensity",
        "max_intensity",
        "median_intensity",
    ]

    X = df[features].values
    y = (df["class"] == "dead").astype(int).values
    # len(images)-fold cross-validation (by image)
    kfold_results = {name: [] for name in models.keys()}

    # Combined df has multiple rows per image, so get unique images,
    # so the train/test separation is at the image level.
    unique_images = df["filename"].unique()
    kf = KFold(n_splits=len(unique_images), shuffle=True, random_state=42)
    sample_indices = np.arange(len(unique_images))
    test_images_all_folds = []

    for fold, (train_idx, test_idx) in enumerate(kf.split(sample_indices)):

        train_images = unique_images[train_idx]
        test_images = unique_images[test_idx]

        train_mask = df["filename"].isin(train_images)
        test_mask = df["filename"].isin(test_images)

        X_train, y_train = X[train_mask], y[train_mask]
        X_test, y_test = X[test_mask], y[test_mask]

        model_filename = pathlib.Path(output_dir) / f"run_{run}_fold_{fold}_model.onnx"
        fold_results = train_and_evaluate_oocyst_classification_models(
            models,
            X_train,
            y_train,
            X_test,
            y_test,
            model_filename=model_filename,
        )
        for model_name in kfold_results:
            kfold_results[model_name].append(fold_results[model_name])

        test_images_all_folds.append(test_images.tolist())

    return test_images_all_folds, kfold_results


def save_kfold_results_to_csv(
    model_results: Dict[str, pd.DataFrame], output_dir: str, run: int
):
    """
    Save K-fold results for each model to CSV files in the output directory.

    Args:
        model_results (Dict[str, pd.DataFrame]): Dictionary of model results DataFrames
        output_dir (str): Directory to save CSV files
        run (int): Run number
    """
    for model_name, results in model_results.items():
        df = pd.DataFrame(results)
        output_csv_path = (
            pathlib.Path(output_dir)
            / f"{model_name.replace(' ', '_').lower()}_run_{run}_kfold_results.csv"
        )
        df.to_csv(output_csv_path, index=False)


def get_best_model_for_run(test_images: list, kfold_results: Dict) -> Tuple[str, int]:
    """
    Determine the best model for a given run based on AUC performance across folds.
    In case of ties, select the model with the lowest dead_f1 variance.

    Args:
        test_images (list): List of test image filenames for each fold
        kfold_results (Dict): Dictionary containing results for each model across folds for a given run

    Returns:
        Tuple[str, int]: Best model name and the number of folds it won
    """

    best_models_all_folds = []
    # Collect results and determine best model per fold
    for fold in range(len(next(iter(kfold_results.values())))):
        # Collect AUC for each model
        aucs = {}
        for model_name in kfold_results.keys():
            aucs[model_name] = kfold_results[model_name][fold]["auc"]

        # Find best model(s) for this fold
        max_auc = max(aucs.values())
        best_model_per_fold = [model for model, auc in aucs.items() if auc == max_auc]
        best_models_all_folds.append(best_model_per_fold)

        print(
            f"Test image(s) {test_images[fold]}: Best model(s) - {', '.join(best_model_per_fold)} (AUC: {max_auc:.4f})"
        )

    # Count wins per model
    model_wins = {model_name: 0 for model_name in kfold_results.keys()}
    for fold_winners in best_models_all_folds:
        for model in fold_winners:
            model_wins[model] += 1

    folds_won = max(model_wins.values())
    models_selected = [model for model, wins in model_wins.items() if wins == folds_won]

    # If there are multiple tied models, select model based that has least dead_f1 variance
    if len(models_selected) == 1:
        print(
            f"Best model within this run: {models_selected[0]} (won {folds_won} folds)"
        )
        return models_selected[0], folds_won
    else:
        # Calculate dead_f1 variance for tied models to determine best generalization
        model_variance = {}
        for model in models_selected:
            dead_f1 = kfold_results[model]["dead_f1"].values
            model_variance[model] = np.var(dead_f1)

        model_selected_with_least_f1variance = min(
            model_variance.items(), key=lambda x: x[1]
        )
        print(
            f"Best model within this run with least variance in dead_f1 score: \
            {model_selected_with_least_f1variance[0]} (variance: {model_selected_with_least_f1variance[1]:.6f})"
        )

        return model_selected_with_least_f1variance[0], folds_won


def main():
    # Dictionary of classifiers from which we will select the best model
    models = {
        "Random Forest": RandomForestClassifier(
            n_estimators=100, class_weight="balanced"
        ),
        # SVM requires a slightly different instantiation, using a pipeline to include scaling
        # which is then also applied during inference. Setting probability to True allows us
        # to treat the output the same as for the other models.
        "SVM": Pipeline(
            [
                ("scaler", StandardScaler()),
                ("classifier", SVC(probability=True, class_weight="balanced")),
            ]
        ),
        "Gradient Boosting": GradientBoostingClassifier(n_estimators=100),
    }
    parser = argparse.ArgumentParser(
        description="Perform model selection for classifying live and dead oocysts using\
                    multiple machine learning algorithms"
    )
    parser.add_argument(
        "input_csv_path",
        type=lambda x: utils.csv_path(
            x, required_columns=["file", "live_oocyst_seg_file", "dead_oocyst_seg_file"]
        ),
        help="Input CSV path containing columns titled 'file', 'live_oocyst_seg_file' and 'dead_oocyst_seg_file'.",
    )
    parser.add_argument(
        "output_dir",
        type=utils.dir_path,
        help="Directory to save results of K-fold model evaluation and selection.",
    )
    parser.add_argument(
        "--number_of_runs",
        type=utils.positive_int,
        default=1,
        help="Number of runs for model evaluation and selection.",
    )

    try:
        args = parser.parse_args()

        df = pd.read_csv(args.input_csv_path)

        all_features = []
        for live_file, dead_file, org_file in zip(
            df["live_oocyst_seg_file"], df["dead_oocyst_seg_file"], df["file"]
        ):

            features_and_class = utils.compute_features_and_class(
                live_file, dead_file, org_file
            )

            all_features.append(features_and_class)

        combined_df = pd.concat(all_features, ignore_index=True)
        combined_df = combined_df[
            ["filename"] + [col for col in combined_df.columns if col != "filename"]
        ]

        selected_models = {"run": [], "best_model": [], "wins": []}

        # Use a temporary directory to store intermediate results across runs and folds
        # in the end select the best model and copy to the output directory
        with tempfile.TemporaryDirectory() as temp_dir:
            for run in range(args.number_of_runs):
                test_images_all_folds, model_parameters_metrics_for_all_folds = (
                    get_all_fold_results(models, combined_df, temp_dir, run)
                )

                save_kfold_results_to_csv(
                    model_parameters_metrics_for_all_folds, temp_dir, run
                )

                model, folds_won = get_best_model_for_run(
                    test_images_all_folds, model_parameters_metrics_for_all_folds
                )

            selected_models["run"].append(run)
            selected_models["best_model"].append(model)
            selected_models["wins"].append(folds_won)

            selected_models_for_all_runs_df = pd.DataFrame(selected_models)
            selected_models_for_all_runs_df.to_csv(
                pathlib.Path(temp_dir) / "selected_models_across_runs.csv", index=False
            )

            best_model_name = selected_models_for_all_runs_df["best_model"].mode()[
                0
            ]  # Most frequently selected model. If tied, select any random one.
            print(f"Most frequently selected model across runs: {best_model_name}")

            selected_model_index_run = selected_models_for_all_runs_df[
                selected_models_for_all_runs_df["best_model"] == best_model_name
            ].index[0]
            run = selected_models_for_all_runs_df.loc[selected_model_index_run, "run"]

            csv_path_for_extracting_best_model = (
                pathlib.Path(temp_dir)
                / f"{best_model_name.replace(' ', '_').lower()}_run_{run}_kfold_results.csv"
            )

            df = pd.read_csv(csv_path_for_extracting_best_model)

            # best_model_name is the algorithm name, e.g. "Random Forest".
            # select the run_0_fold_0 parameters as the best model parameters
            # selecting the run and fold with best AUC does not necessarily yield
            # better results on other datasets as the test image may have been easier in
            # that particular instance.
            final_model_path = (
                pathlib.Path(temp_dir)
                / f"{best_model_name.replace(' ', '_').lower()}_run_0_fold_0_model.onnx"
            )

            shutil.copy2(
                final_model_path,
                pathlib.Path(args.output_dir)
                / f"{best_model_name.replace(' ', '_').lower()}.onnx",
            )

        return 0

    except Exception as e:
        print(f"Error during model selection: {e}")
        return 1


if __name__ == "__main__":
    sys.exit(main())
