# -*- coding: utf-8 -*-
# -----------------------------------------------------------------------------
# Author: Synne Krekling Lien
# Contact: synne.lien@sintef.no
# Date: 24.02.2026
# Repository: https://github.com/synnekreklinglien/COF-tool
#
# Supervised classification of building category and heating type using
# hourly electricity smart meter features and outdoor temperature.
#
# Functions to train, evaluate, and apply classification models.
# -----------------------------------------------------------------------------

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib.colors import PowerNorm
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.preprocessing import LabelEncoder

from . import CofClassifyFeatureExtraction as CofClassifyFeatureExtraction

print("Imported COF-Tool CofClassifyClassify.")

@dataclass
class ModelBundle:
    """
    Data container for the trained building classification model.

    Stores the fitted classifier and the fitted label encoder used to map
    between string class labels (here "<building_category>_<heating_type>") and
    the integer labels used by the model during training. This bundle is used
    during inference and evaluation to ensure consistent label handling.

    Parameters
    ----------
    model : object
        Fitted classification model (default RandomForestClassifier). 
    label_encoder : sklearn.preprocessing.LabelEncoder
        Fitted label encoder used during training to encode and decode class
        labels.
    """
    model: object
    label_encoder: LabelEncoder


DEFAULT_MODEL_FILE = "classification_model.joblib"


def _resources_dir():
    """
    Return the path to the package resources directory.
    This directory contains trained classification models and other files
    required at inference time.

    Returns
    -------
    pathlib.Path
        Path to the `resources` directory.
    """
    return Path(__file__).resolve().parent.parent / "resources"


def _default_model_path():
    """
    Return the default path to the trained default classification model.

    This path points to the default model file stored in the package resources
    directory and is used when no custom model path is provided.

    Returns
    -------
    pathlib.Path
        Path to the default trained classification model file.
    """
    return _resources_dir() / DEFAULT_MODEL_FILE


def save_model_to_file(bundle, path):
    """
    Save a trained classification model bundle to disk.

    Writes a `ModelBundle` containing the fitted classifier and its corresponding
    label encoder to a file using joblib. The saved file can later be loaded for
    classificatin of building type without retraining the model.

    Parameters
    ----------
    bundle : ModelBundle
        Fitted classification model (default RandomForestClassifier). 
    path : str or pathlib.Path
        Target file path where the model bundle is saved.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(bundle, path)


def load_model(path=None):
    """
    Load a trained classification model bundle from disk.

    Loads a previously saved `ModelBundle` containing a trained classifier and
    its corresponding label encoder. If no path is provided, the default model
    stored in the package resources directory is used.

    Parameters
    ----------
    path : str or pathlib.Path, optional
        Path to a saved model bundle. If None, the default model path is used.

    Returns
    -------
    ModelBundle
        Loaded model bundle containing the classifier and label encoder.

    """
    load_path = Path(path) if path is not None else _default_model_path()

    if not load_path.exists():
        raise FileNotFoundError(
            f"No saved model found at {load_path}. Default is {_default_model_path()}."
        )

    bundle = joblib.load(load_path)
    if not hasattr(bundle, "model") or not hasattr(bundle, "label_encoder"):
        raise TypeError("Loaded object does not look like a ModelBundle.")

    return bundle


def _get_expected_feature_names(model):
    """
    Retrieve the feature order expected by the trained classification model.

    Parameters
    ----------
    model : object
        Trained classification model 

    Returns
    -------
    names : list of str or None
        Feature names if available, else None
    """
    if hasattr(model, "feature_names_in_"):
        return list(model.feature_names_in_)

    if hasattr(model, "named_steps"):
        for step in reversed(list(model.named_steps.values())):
            if hasattr(step, "feature_names_in_"):
                return list(step.feature_names_in_)

    return None


def _align_X_to_expected(X, expected):
    """
    Align building classification features to the input expected by the trained
    RandomForest model.

    Ensures that feature columns are in the same order as used during training,
    which is required for correct predictions. As a safety measure, missing
    expected features are added with value 0 and extra features are removed.

    Parameters
    ----------
    X : pandas.DataFrame
        Extracted building-level feature data.
    expected : list of str
        Feature names in the order used during model training.

    Returns
    -------
    pandas.DataFrame
        Feature DataFrame aligned to the model input.
    """
    X = X.copy()

    missing = [c for c in expected if c not in X.columns]
    for c in missing:
        X[c] = 0

    return X.loc[:, expected]



def train_classification_model(X_train, y_train, model=None, save_model=True, filename=None):    
    """
    Train a building classification model for building category and heating type.

    Trains a supervised classifier (by default a RandomForestClassifier) using
    building-level feature vectors extracted from electricity time series and
    outdoor temperature data (see "CofClassificationFeatureExtraction". 
    The target labels represent the true buildingcategory and space-heating type 
    (e.g. "Apt_EH", "SFH_NEH").

    The function also fits a LabelEncoder to map string labels to integer class
    indices and optionally saves the trained model and encoder as a single
    bundle for later use when classifying new buildings.

    Parameters
    ----------
    X_train : array-like or pandas.DataFrame
        Training feature matrix containing extracted building features
        (fixed-size vectors, ~180 features per building).
    y_train : array-like
        Ground-truth building labels used for training.
    model : object, optional
        scikit-learn compatible classifier. If None, a
        RandomForestClassifier(random_state=42) is used.
    save_model : bool, optional
        Whether to save the trained model bundle to disk (default True).
    filename : str or pathlib.Path, optional
        Target file path for saving the trained model. If None, the default
        model path is used.

    Returns
    -------
    model : object
        Trained classification model.
    label_encoder : sklearn.preprocessing.LabelEncoder
        Fitted label encoder used to encode and decode building labels.
    """
    if model is None:
        model = RandomForestClassifier(random_state=42)

    label_encoder = LabelEncoder()
    y_train_encoded = label_encoder.fit_transform(y_train)

    model.fit(X_train, y_train_encoded)

    if save_model:
        save_path = Path(filename) if filename is not None else _default_model_path()
        save_model_to_file(ModelBundle(model=model, label_encoder=label_encoder), save_path)

    return model, label_encoder


def evaluate_classification_model(model, label_encoder, X_val, y_val, plot=True):
    """
    Evaluate a trained building classification model on validation data.

    Evaluates a classifier trained to predict building category and heating type
    using extracted building-level feature vectors. Validation labels are decoded
    back to their original string form to produce an interpretable classification
    report. Optionally, a confusion matrix is visualized to show class-level
    performance.

    Parameters
    ----------
    model : object
        Trained classification model (e.g. RandomForestClassifier).
    label_encoder : LabelEncoder
        Label encoder fitted during training.
    X_val : array-like or pandas.DataFrame
        Validation feature matrix containing extracted building features.
    y_val : array-like
        Ground-truth validation labels.
    plot : bool, optional
        Whether to plot a confusion matrix heatmap (default True).

    Returns
    -------
    str
        Text classification report with decoded class labels.
    """
    y_val_encoded = label_encoder.transform(y_val)
    y_val_pred = model.predict(X_val)

    y_val_actual = label_encoder.inverse_transform(y_val_encoded)
    y_val_predicted = label_encoder.inverse_transform(y_val_pred)

    report = classification_report(y_val_actual, y_val_predicted)
    print(report)

    if plot:
        classes = label_encoder.classes_
        cm = confusion_matrix(y_val_actual, y_val_predicted, labels=classes)

        plt.figure(figsize=(6, 5))
        sns.heatmap(
            cm,
            annot=True,
            fmt="d",
            xticklabels=classes,
            yticklabels=classes,
            cmap="Blues",
            cbar=False,
            annot_kws={"color": "black"},
            norm=PowerNorm(gamma=0.2),
        )
        plt.xlabel("Predicted")
        plt.ylabel("Actual")
        plt.title(type(model).__name__)
        plt.tight_layout()
        plt.show()

    return report


def predict_class_probabilities(X_test, model=None, label_encoder=None, model_path=None):
    """
    Prediction function for building classification.
    
    Predicts class probabilities for each possible building class
    (combination of building category and heating type, e.g. "Apt_EH",
    "Off_NEH") for a single building, based on its extracted feature vector
    and a trained classification model.
    
    If `model` and `label_encoder` are not provided, and no `model_path` is
    given, the saved default trained model bundle (RandomForest classifier)
    is loaded from disk.
    
    Parameters
    ----------
    X_test : array-like or pandas.DataFrame
        Extracted feature vector for a single building.
    model : object, optional
        Trained classification model.
    label_encoder : LabelEncoder, optional
        Label encoder fitted during training.
    model_path : str or Path, optional
        Path to a saved model bundle. If None, the default model path is used.
    
    Returns
    -------
    pandas.DataFrame
        Single-row table where each column corresponds to a building class and
        each value is the predicted probability for that class.
    """

    if model is None or label_encoder is None:
        bundle = load_model(model_path)
        model = bundle.model
        label_encoder = bundle.label_encoder

    X_use = X_test

    if isinstance(X_test, pd.DataFrame):
        expected = _get_expected_feature_names(model)
        if expected is not None:
            X_use = _align_X_to_expected(X_test, expected)
        else:
            X_use = X_test.reindex(sorted(X_test.columns), axis=1)

    probs = model.predict_proba(X_use)
    class_labels = label_encoder.inverse_transform(np.arange(len(label_encoder.classes_)))

    if isinstance(X_test, pd.DataFrame):
        return pd.DataFrame(probs, columns=class_labels, index=X_test.index)

    return pd.DataFrame(probs, columns=class_labels)


def predict_class(X_test, model=None, label_encoder=None, model_path=None):
    """
    Prediction function for building classification.

    Predicts the most likely building class (category and heating type) for a
    single building by selecting the class with the highest probability from
    `predict_class_probabilities`.

    Model loading follows the same logic as in `predict_class_probabilities`.

    Parameters
    ----------
    X_test : array-like or pandas.DataFrame
        Extracted feature vector for a single building.
    model : object, optional
        Trained classification model.
    label_encoder : LabelEncoder, optional
        Label encoder fitted during training.
    model_path : str or Path, optional
        Path to a saved model bundle.

    Returns
    -------
    pandas.Series
        Predicted building class label.
    """
    
    probs = predict_class_probabilities(X_test, model=model, label_encoder=label_encoder, model_path=model_path)
    return probs.idxmax(axis=1)


def predict_class_probabilities_ts(ts, building_id="building", model=None, label_encoder=None, model_path=None):
    """
    Prediction function for building classification from a building time series
    (not feature-extracted).
    
    Extracts the feature vector for a single building time series using the feature
    extraction function `CofClassifyFeatureExtraction.get_extracted_features_from_building_ts`,
    and then calls `predict_class_probabilities` to compute class probabilities.
    
    Model loading follows the same logic as in `predict_class_probabilities`.
    
    Parameters
    ----------
    ts : object
        Building time series in TREASURE format.
    building_id : str, optional
        Building identifier used in prints.
    model : object, optional
        Trained classification model.
    label_encoder : LabelEncoder, optional
        Label encoder fitted during training.
    model_path : str or Path, optional
        Path to a saved model bundle.
    
    Returns
    -------
    pandas.DataFrame
        Single-row table of class probabilities for the building.
    """
    print(f"Predicting class probabilities for {building_id}")

    building_features = CofClassifyFeatureExtraction.get_extracted_features_from_building_ts(
        ts, building_id=building_id
    )

    building_probs = predict_class_probabilities(
        building_features, model=model, label_encoder=label_encoder, model_path=model_path
    )

    print("Done.")
    return building_probs


def predict_class_ts(ts, building_id="building", model=None, label_encoder=None, model_path=None):
    """
    Prediction function for building classification from a building time series
    (not feature extracted),

    Extracts the feature vector for a single building time series using
    `get_extracted_features_from_building_ts` and calls `predict_class` to return
    the most likely building class.

    Model loading follows the same logic as in `predict_class_probabilities`.

    Parameters
    ----------
    ts : object
        Building time series in TREASURE format.
    building_id : str, optional
        Building identifier used in prints.
    model : object, optional
        Trained classification model.
    label_encoder : LabelEncoder, optional
        Label encoder fitted during training.
    model_path : str or Path, optional
        Path to a saved model bundle.

    Returns
    -------
    str
        Predicted building class label.
    """
    print(f"Predicting class for {building_id}")

    building_features = CofClassifyFeatureExtraction.get_extracted_features_from_building_ts(
        ts, building_id=building_id
    )

    building_class = predict_class(
        building_features, model=model, label_encoder=label_encoder, model_path=model_path
    )

    print("Done")
    return str(building_class.iloc[0])


def plot_building_class_probabilities(building_class_probabilities, top_n=3, title="Building class probabilities"):
    """
    Visualization function for building classification results.

    Plots a pie chart of predicted building class probabilities for a single
    building, as returned by `predict_class_probabilities` or
    `predict_class_probabilities_ts`. The most likely classes are shown
    explicitly, while remaining classes can be grouped as "Other".

    Parameters
    ----------
    building_class_probabilities : pandas.DataFrame
        Single-row DataFrame of class probabilities for one building.
    top_n : int, optional
        Number of most probable classes to display explicitly (default 3).
        Remaining classes are grouped as "Other".
    title : str, optional
        Title of the plot.

    Returns
    -------
    matplotlib.figure.Figure
        Figure object containing the pie chart.
    """
    if building_class_probabilities.shape[0] != 1:
        raise ValueError("Dataframe must have exactly one row")

    probs = building_class_probabilities.iloc[0]

    total = probs.sum()
    if total != 0:
        probs = probs / total

    probs = probs.sort_values(ascending=False)

    if len(probs) > top_n:
        top = probs.iloc[:top_n].copy()
        other_sum = probs.iloc[top_n:].sum()
        if other_sum > 0:
            top["Other"] = other_sum
        plot_probs = top
    else:
        plot_probs = probs

    labels = plot_probs.index.tolist()
    values = plot_probs.values

    cmap = plt.get_cmap("tab20")
    colors = [cmap(i) for i in range(len(values))]

    fig, ax = plt.subplots(figsize=(5, 4))
    ax.pie(
        values,
        labels=labels,
        colors=colors,
        startangle=90,
        autopct="%1.0f %%",
        wedgeprops=dict(edgecolor="white"),
        textprops=dict(fontsize=11),
    )

    ax.set_title(title, fontsize=11, pad=10)
    ax.axis("equal")
    return fig