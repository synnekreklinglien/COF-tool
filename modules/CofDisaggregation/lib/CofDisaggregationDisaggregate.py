# -*- coding: utf-8 -*-
# -----------------------------------------------------------------------------
# Author: Synne Krekling Lien
# Contact: synne.lien@sintef.no
# Date: 23.02.2026
# Repository: https://github.com/synnekreklinglien/COF-tool
#
# Estimation (disaggregation) of electricity use for heating purposes from
# building time series data with hourly smart meter readings and outdoor 
# temperature.
# Functions: train, load, apply, and evaluate disaggregation models.
# -----------------------------------------------------------------------------

from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from catboost import CatBoostRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from . import CofDisaggregationFeatureExtraction as CofDisaggregateFeatureExtraction

print("Imported COF-Tool CofDisaggregationDisaggregate.")


def load_disaggregation_model(model_name="model", savedir=None):
    """
    Load a trained CatBoost disaggregation model from disk.
 
    Loads a previously saved/trained CatBoost regression model used for electric heating
    disaggregation. The model is expected to be stored in CatBoost's native
    `.cbm` format inside the specified directory.
 
    Parameters
    ----------
    model_name : str, optional
        Name of the model file without extension. Defaults to "model".
    savedir : str or pathlib.Path
        Directory containing the saved model file.
 
    Returns
    -------
    CatBoostRegressor
        Loaded CatBoost regression model.
 
    Raises
    ------
    ValueError
        If `savedir` is not provided.
 
    """
    if savedir is None:
        raise ValueError("savedir must be provided")

    savedir = Path(savedir)
    filepath = savedir / f"{model_name}.cbm"

    model = CatBoostRegressor()
    model.load_model(str(filepath))
    return model


def _load_default_model():
    """
    Load the default disaggregation model from the module resources.
    
    Loads the pre-trained CatBoost regression model bundled with the module,
    typically stored in `Cof-Tool/modules/CofDisaggregation/resources`. 
    The trained default model is intended for use when no user-specified model path is provided.
    
    Returns
    -------
    CatBoostRegressor
        Loaded trained CatBoost regression model for electric heating disaggregation. 
    
    """
    base_dir = Path(__file__).resolve().parent.parent
    model_dir = base_dir / "resources"
    return load_disaggregation_model(
        model_name="disaggregation_model_All_AF_without_NEH",
        savedir=model_dir,
    )


def train_disaggregation_model(train_X, train_y, model_name="model", savedir=None):
    """
    Train a CatBoost disaggregation model for heating electricity estimation.
    
    Trains a supervised CatBoost regression model to disaggregate total
    hourly electricity/energy use into electricity use for heating and other loads. 
    The model is trained on feature-extracted building-level datasets generated with the
    CofDisaggregationFeatureExtraction module.
    
    Input data (features) may be provided either as a single pandas DataFrame representing
    one building, or as a dictionary of DataFrames representing multiple buildings.
    In the multi-building case, all datasets are concatenated prior to training.
    
    Parameters
    ----------
    train_X : pandas.DataFrame or dict
        Feature-extracted training data for one building, or a dictionary of
        DataFrames for multiple buildings.
    train_y : pandas.Series, pandas.DataFrame, or dict
        Target heating electricity consumption for one building, or a dictionary
        of Series/DataFrames for multiple buildings.
    model_name : str, optional
        Model name used when saving. Defaults to "model".
    savedir : str or pathlib.Path, optional
        Directory where the trained model is saved. If None, the model is not saved.
    
    Returns
    -------
    CatBoostRegressor
        Trained CatBoost regression model for electric heating disaggregation.
    
    """
    print("Training model.")

    if isinstance(train_X, dict):
        train_X = pd.concat(list(train_X.values()), axis=0)

    if isinstance(train_y, dict):
        train_y = pd.concat(list(train_y.values()), axis=0)

    model = CatBoostRegressor(verbose=0, random_seed=42)
    model.fit(train_X, train_y)

    if savedir is not None:
        savedir = Path(savedir)
        savedir.mkdir(parents=True, exist_ok=True)
        filepath = savedir / f"{model_name}.cbm"
        model.save_model(str(filepath))

    return model


def plot_disaggregation(X, preds, bname="Building"):
    """
    Plot total electricity/energy use and estimate electricity/energy use for heating.
    
    Creates a time-series plot comparing the measured total electricity/energy use with the
    estimated electricity use for heating produced by a disaggregation model.
    
    Parameters
    ----------
    X : array-like
        Time series of total electricity/energy use.
    preds : array-like
        Time series of estimated electricity use for heating.
    bname : str, optional
        Building name used in the plot title. Defaults to "Building".
    
    Returns
    -------
    matplotlib.figure.Figure
        Matplotlib figure object.

    """
    x = range(len(X))

    fig, ax = plt.subplots(figsize=(8, 4))
    ax.plot(x, X, linewidth=0.8, label="Total measured energy use")
    ax.plot(x, preds, linestyle=":", linewidth=0.8, label="Estimated energy use for heating")

    ax.set_xlabel("N hours")
    ax.set_ylabel("Energy use [Wh/h]")
    ax.set_title(f"Total vs Estimated Heating Energy Use for {bname}")

    ax.legend(loc="center left", bbox_to_anchor=(1.02, 0.5), frameon=False)

    fig.tight_layout()
    plt.show()

    return fig


def predict_disaggregation(X_df, model):
    """
    Predict electricity/energy use for heating from feature-extracted input data.
    
    Estimates electricity/energy use for heating for a single building based on
    feature-extracted input data and a trained disaggregation model. This function
    assumes that a model is already available and applies no model-loading logic.
    
    Post-processing is applied to ensure physically consistent predictions:
    
    - negative predictions are clipped to zero
    - NaN values are preserved where the total electricity/energy use is NaN
    - predictions are forced to zero where the total electricity/energy use is zero
    
    Parameters
    ----------
    X_df : pandas.DataFrame
        Feature-extracted input data for a single building, generated with
        CofDisaggregationFeatureExtraction. The DataFrame must contain a column
        named "X" representing total electricity/energy use.
    model : CatBoostRegressor
        Trained disaggregation model.
    
    Returns
    -------
    numpy.ndarray
        Predicted electricity/energy use for heating time series.
    
    """

    preds = np.asarray(model.predict(X_df), dtype=float)

    if preds.ndim == 2 and preds.shape[0] == 1:
        preds = preds[0]
    elif preds.ndim == 2 and preds.shape[1] == 1:
        preds = preds[:, 0]

    X = X_df["X"].to_numpy(dtype=float, copy=False)

    preds = np.maximum(preds, 0.0)
    preds[np.isnan(X)] = np.nan
    preds[X == 0.0] = 0.0

    return preds


def disaggregate_feature_extracted_building(X_df, model=None):
    """
    Disaggregate electricity/energy use for heating for a feature-extracted building.
    
    Disaggregates electricity/energy use for heating for a single building where
    features have already been extracted. If no model is provided, the default
    trained disaggregation model from the repository is used.
    
    Parameters
    ----------
    X_df : pandas.DataFrame
        Feature-extracted input data for a single building. The DataFrame must
        contain a column named "X" representing total electricity/energy use.
    model : CatBoostRegressor, optional
        Trained disaggregation model. If None, the default model is loaded.
    
    Returns
    -------
    numpy.ndarray
        Predicted electricity/energy use for heating time series.
    
    """
    if model is None:
        model = _load_default_model()
    return predict_disaggregation(X_df, model)


def disaggregate_trea_ts(ts, model=None, forced_X=None):
    """
    Disaggregate electricity/energy use for heating from TREASURE-format building data.
    
    Extracts disaggregation features from data for an individual building provided
    in TREASURE format and estimates electricity/energy use for heating using a
    trained disaggregation model. Feature extraction is performed internally
    before applying the disaggregation model.
    
    Parameters
    ----------
    ts : pandas.DataFrame
        Data for an individual building in TREASURE format.
    model : CatBoostRegressor, optional
        Trained disaggregation model. If None, the default model is loaded.
    forced_X : array-like, optional
        If provided, this series is used as the total electricity/energy use
        instead of the value extracted from the input TREASURE data.
    
    Returns
    -------
    numpy.ndarray
        Predicted electricity/energy use for heating time series.
        
    """
    X_df = CofDisaggregateFeatureExtraction.get_extracted_features_from_building_ts(
        ts,
        None,
        None,
        forced_X,
    )
    return disaggregate_feature_extracted_building(X_df, model)


def evaluate_disaggregation(true_Y, pred_Y, make_plots=False, building_id="Building"):
    """
    Evaluate electricity/energy disaggregation performance for heating.
    
    Evaluates disaggregation performance by comparing measured and predicted
    electricity/energy use for heating for an individual building. Standard
    performance metrics are computed, and optional diagnostic plots of the
    measured and predicted time series and their pointwise comparison can be
    generated.
    
    Parameters
    ----------
    true_Y : array-like
        Measured electricity/energy use for heating.
    pred_Y : array-like
        Predicted electricity/energy use for heating.
    make_plots : bool, optional
        If True, generate time series and scatter plots comparing measured and
        predicted values. Defaults to False.
    building_id : str, optional
        Identifier for the individual building, used in plot titles.
    
    Returns
    -------
    dict
        Dictionary containing the following performance metrics:
    
        - R2 : coefficient of determination
        - CV_RMSE : coefficient of variation of the root mean squared error
        - NMAE : normalized mean absolute error
        - NMBE : normalized mean bias error
    
    """

    true_Y = np.asarray(true_Y, dtype=float).ravel()
    pred_Y = np.asarray(pred_Y, dtype=float).ravel()

    r2 = r2_score(true_Y, pred_Y)
    rmse = np.sqrt(mean_squared_error(true_Y, pred_Y))
    mean_true = np.mean(true_Y)

    cv_rmse = rmse / mean_true
    nmae = mean_absolute_error(true_Y, pred_Y) / mean_true
    nmbe = np.sum(pred_Y - true_Y) / (len(true_Y) * mean_true)

    performance = {
        "R2": r2,
        "CV_RMSE": cv_rmse,
        "NMAE": nmae,
        "NMBE": nmbe,
    }

    if make_plots:
        plt.figure(figsize=(10, 4))
        plt.plot(true_Y, label="Measured", alpha=0.6)
        plt.plot(pred_Y, label="Predicted", alpha=0.6)
        plt.xlabel("N Timesteps")
        plt.ylabel("Energy use [Wh/h]")
        plt.title(f"Measured vs predicted electricity for heating for {building_id}")
        plt.legend()
        plt.tight_layout()
        plt.show()

        plt.figure(figsize=(5, 5))
        plt.scatter(true_Y, pred_Y)
        plt.xlabel("Measured")
        plt.ylabel("Predicted")
        plt.title(f"Measured vs predicted electricity for heating [Wh/h] for {building_id}")
        plt.tight_layout()
        plt.show()

    return performance