import argparse
import numpy as np
import SimpleITK as sitk
import pandas as pd
from typing import Tuple
import pathlib
from sklearn.metrics import (
    roc_curve,
)
import urllib.request


def compute_label_features(
    label_image: sitk.Image, intensity_image: sitk.Image
) -> pd.DataFrame:
    """
    Compute features such as area, perimeter, elongation, flatness, mean_intensity, min_intensity,
    max_intensity, and median_intensity from labeled regions in an image using SimpleITK.

    Args:
        label_image (sitk.Image): Multi label segmentation image
        intensity_image (sitk.Image): Source intensity image for feature extraction

    Returns:
        list: Feature dictionaries for each of the objects in the label image
    """

    label_intensity_filter = sitk.LabelIntensityStatisticsImageFilter()
    # Potentially normalize the intensity image to be more robust to variations in acquisition
    # label_intensity_filter.Execute(label_image, sitk.Normalize(intensity_image))
    label_intensity_filter.Execute(label_image, intensity_image)

    object_indexes = label_intensity_filter.GetLabels()

    features = []

    for object_index in object_indexes:
        feature_dict = {
            "area": label_intensity_filter.GetNumberOfPixels(object_index),
            "perimeter": label_intensity_filter.GetPerimeter(object_index),
            "elongation": label_intensity_filter.GetElongation(
                object_index
            ),  # Elongation measures how stretched or elongated an object is (ratio of major to minor axis lengths);
            # higher values mean more elongated.
            "flatness": label_intensity_filter.GetFlatness(
                object_index
            ),  # Flatness measures how "flat" (disk-like) the object is (ratio of smallest to largest principal axis);
            # values close to 1 are flat, lower values are more spherical.
            "mean_intensity": label_intensity_filter.GetMean(object_index),
            "min_intensity": label_intensity_filter.GetMinimum(object_index),
            "max_intensity": label_intensity_filter.GetMaximum(object_index),
            "median_intensity": label_intensity_filter.GetMedian(object_index),
        }

        features.append((object_index, feature_dict))

    return features


def compute_features_and_class(
    live_oocyst_segmentation_file: str,
    dead_oocyst_segmentation_file: str,
    image_intensity_file: str,
) -> pd.DataFrame:
    """
    Process a image to extract features using it's corresponding live and dead oocyst segmentation files.

    Args:
        live_oocyst_segmentation_file (str): Path to live oocyst multi label segmentation file
        dead_oocyst_segmentation_file (str): Path to dead oocyst multi label segmentation file
        image_intensity_file (str): Path to original image

    Returns:
        pd.DataFrame: Combined features with labels
    """

    live_img = sitk.ReadImage(live_oocyst_segmentation_file)
    dead_img = sitk.ReadImage(dead_oocyst_segmentation_file)
    org_img = sitk.ReadImage(image_intensity_file)

    live_features = compute_label_features(live_img, org_img)
    live_features_df = pd.DataFrame([live_feature[1] for live_feature in live_features])

    dead_features = compute_label_features(dead_img, org_img)
    dead_features_df = pd.DataFrame([dead_feature[1] for dead_feature in dead_features])

    live_features_df["class"] = "live"
    dead_features_df["class"] = "dead"

    combined_features = pd.concat(
        [live_features_df, dead_features_df], ignore_index=True
    )
    combined_features["filename"] = image_intensity_file

    return combined_features


def calculate_youden_index(
    y_true: np.ndarray, y_scores: np.ndarray
) -> Tuple[float, float]:
    """
    Calculate optimal threshold using Youden's index.

    Args:
        y_true (np.ndarray): True labels
        y_scores (np.ndarray): Predicted probabilities

    Returns:
        optimal_threshold (float): Threshold that maximizes Youden's index
    """
    fpr, tpr, thresholds = roc_curve(y_true, y_scores)
    youden_index = tpr - fpr
    optimal_idx = np.argmax(youden_index)
    optimal_threshold = thresholds[optimal_idx]

    return optimal_threshold


def csv_path(path, required_columns={}):
    """
    Define the csv_path type for use with argparse. Checks
    that the given path string is a path to a csv file and that the
    header of the csv file contains the required columns.
    """
    p = pathlib.Path(path)
    required_columns = set(required_columns)
    if p.is_file():
        try:  # only read the csv header
            expected_columns_exist = required_columns.issubset(
                set(pd.read_csv(path, nrows=0).columns.tolist())
            )
            if expected_columns_exist:
                return p
            else:
                raise argparse.ArgumentTypeError(
                    f"Invalid argument ({path}), does not contain all expected columns."
                )
        except UnicodeDecodeError:
            raise argparse.ArgumentTypeError(
                f"Invalid argument ({path}), not a csv file."
            )
    else:
        raise argparse.ArgumentTypeError(f"Invalid argument ({path}), not a file.")


def dir_path(path):
    """
    Define the dir_path type for use with argparse. Checks
    that the given path string is a path to an existing directory.
    """
    p = pathlib.Path(path)
    if p.is_dir():
        return p
    else:
        raise argparse.ArgumentTypeError(
            f"Invalid argument ({path}), not a directory path or directory does not exist."
        )


def positive_int(i):
    """
    Define the positive_int type for use with argparse. Checks
    that the given input is an integer greater than zero (Python
    doesn't have an unsigned int type).
    """
    res = int(i)
    if res > 0:
        return res
    else:
        raise argparse.ArgumentTypeError(
            f"Invalid argument ({i}), expected value > 0 ."
        )


def path_to_remote_file(local_file_path, remote_url):
    """
    Check if a local file path exists; if does not, download it from a remote URL.
    Args:
        local_file_path (str): Path to the local file
        remote_url (str): URL of the remote file to download if local file doesn't exist
    """
    if not local_file_path or not pathlib.Path(local_file_path).is_file():
        urllib.request.urlretrieve(remote_url, local_file_path)

    return pathlib.Path(local_file_path)
