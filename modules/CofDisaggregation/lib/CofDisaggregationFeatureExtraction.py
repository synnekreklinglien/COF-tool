# -*- coding: utf-8 -*-
# -----------------------------------------------------------------------------
# Author: Synne Krekling Lien
# Contact: synne.lien@sintef.no
# Date: 23.02.2026
# Repository: https://github.com/synnekreklinglien/COF-tool
#
# Feature extraction from hourly building time series data for
# estimating the electricity for heating based on the hourly
# smart meter measurements (for imported electricity) and outdoor temperature.
# -----------------------------------------------------------------------------

import os
import string
import warnings

import numpy as np
import pandas as pd
import pwlf
import scipy.stats as stats
from scipy.stats import ConstantInputWarning, norm

from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

from ...CofClassification.lib import ReadtreaData as treaData
from ...CofClassification.lib import CofClassifyClassify as CofClassifyClassify

print("Imported COF-Tool CofDisaggregationFeatureExtraction.")


def _as_float32(s):
    """
    Helper function to reduce memory usage of feature data.

    (Attempts to) Convert pandas Series to float32, which is sufficient for the
    extracted building features used in this project and reduces memory
    consumption when handling large feature tables. If conversion is not
    possible, the input Series is returned unchanged.

    Parameters
    ----------
    s : pandas.Series
        Input Series to convert.

    Returns
    -------
    pandas.Series
        Series converted to float32 when possible, otherwise returned as-is.
    """
    if pd.api.types.is_float_dtype(s) and s.dtype == np.float32:
        return s
    try:
        return s.astype(np.float32)
    except Exception:
        return s


def _concat_new_cols(df, new_cols):
    """
    Helper function. Append new columns to the feature DataFrame.
    
    Creates a DataFrame from `new_cols` (aligned to `df.index`) and concatenates
    it column-wise to `df`. If `new_cols` is empty, `df` is returned unchanged.
    
    Parameters
    ----------
    df : pandas.DataFrame
        Input DataFrame.
    new_cols : dict
        Mapping of column names to column values.
    
    Returns
    -------
    pandas.DataFrame
        DataFrame with new columns appended.
    """
    if not new_cols:
        return df
    return pd.concat([df, pd.DataFrame(new_cols, index=df.index)], axis=1)


def add_date_features(df):
    """
    Feature extraction function for the final feature set.

    Creates time-related features per time step from the "TimeStamp" column. The function adds
    calendar features describing the current time step (row) in the day, week, month, and year,
    as well as trigonometric representations of the hour of day.

    Seasons are assigned based on the month:
        1 = winter (Dec–Feb),
        2 = spring (Mar–May),
        3 = summer (Jun–Aug),
        4 = autumn (Sep–Nov).

    Parameters
    ----------
    df : pandas.DataFrame
        Final features dataset containing a pandas datetime column named
        "TimeStamp".

    Returns
    -------
    pandas.DataFrame
        Final features dataset with additional time-based feature columns:
        - hour
        - dayofweek
        - month
        - year
        - season
        - weekend
        - sin_hour
        - cos_hour
    """
    seasons = {1: 1, 2: 1, 3: 2, 4: 2, 5: 2, 6: 3, 7: 3, 8: 3, 9: 4, 10: 4, 11: 4, 12: 1}

    ts = df["TimeStamp"]
    df["hour"] = ts.dt.hour.astype("int8", copy=False)
    df["dayofweek"] = ts.dt.dayofweek.astype("int8", copy=False)
    df["month"] = ts.dt.month.astype("int8", copy=False)
    df["year"] = ts.dt.year.astype("int16", copy=False)
    df["season"] = ts.dt.month.map(seasons).astype("int8", copy=False)
    df["weekend"] = (df["dayofweek"] > 4).astype("int8", copy=False)

    hour = df["hour"].to_numpy(dtype=np.float32, copy=False)
    df["sin_hour"] = np.sin(2 * np.pi * hour / 24).astype(np.float32)
    df["cos_hour"] = np.cos(2 * np.pi * hour / 24).astype(np.float32)
    return df


def add_dailyStatisticsFeatures(df, col):
    """
     Feature extraction function for the final feature set.
    
     Computes per-day summary statistics for the specified column and adds them
     as features at every time step. For each day, the same daily statistic value
     is assigned to all rows belonging to that day.
    
     Parameters
     ----------
     df : pandas.DataFrame
         Final features dataset containing a pandas datetime column named
         "TimeStamp" and the specified input column.
     col : str
         Name of the column for which daily statistics are computed. The column
         typically represents total energy use ("X").
    
     Returns
     -------
     pandas.DataFrame
         Final features dataset with additional daily-statistic feature columns,
         added to every time step:
         - daily_mean_{col}
         - daily_max_{col}
         - daily_min_{col}
         - daily_std_{col}
         - daily_var_{col}
     """
    day_key = df["TimeStamp"].dt.date
    g = df.groupby(day_key, sort=False)[col]

    new_cols = {
        f"daily_mean_{col}": _as_float32(g.transform("mean")),
        f"daily_max_{col}": _as_float32(g.transform("max")),
        f"daily_min_{col}": _as_float32(g.transform("min")),
        f"daily_std_{col}": _as_float32(g.transform("std")),
        f"daily_var_{col}": _as_float32(g.transform("var")),
    }
    return _concat_new_cols(df, new_cols)


def add_yearlyStatisticsFeatures(df, col):
    """
    Feature extraction function for the final feature set.

    Computes per-year summary statistics for the specified column and adds them
    as features at every time step. For each year, the same yearly statistic
    value is assigned to all rows belonging to that year.

    Parameters
    ----------
    df : pandas.DataFrame
        Final features dataset containing a column named "year" and the
        specified input column.
    col : str
        Name of the column for which yearly statistics are computed. The column
        typically represents total energy use ("X").

    Returns
    -------
    pandas.DataFrame
        Final features dataset with additional yearly-statistic feature columns,
        added to every time step:
        - yearly_mean_{col}
        - yearly_max_{col}
        - yearly_min_{col}
        - yearly_std_{col}
        - yearly_var_{col}
    """
    g = df.groupby(df["year"], sort=False)[col]

    new_cols = {
        f"yearly_mean_{col}": _as_float32(g.transform("mean")),
        f"yearly_max_{col}": _as_float32(g.transform("max")),
        f"yearly_min_{col}": _as_float32(g.transform("min")),
        f"yearly_std_{col}": _as_float32(g.transform("std")),
        f"yearly_var_{col}": _as_float32(g.transform("var")),
    }
    return _concat_new_cols(df, new_cols)


def add_rollingStatisticsFeatures_multi(df, col, windows):
    """
    Feature extraction function for the final feature set.

    Computes rolling summary statistics for the specified column using multiple
    window sizes and adds them as features at every time step. For each window
    size, the function computes statistics over a trailing window ending at the
    current time step, as well as over a forward window starting at the next
    time step.

    Parameters
    ----------
    df : pandas.DataFrame
        Final features dataset containing the specified input column.
    col : str
        Name of the column for which rolling statistics are computed. The column
        typically represents total energy use ("X").
    windows : iterable
        Iterable of integer window sizes, given in number of time steps.

    Returns
    -------
    pandas.DataFrame
        Final features dataset with additional rolling-statistic feature
        columns, added to every time step. For each window size w, the following
        features are added:
        - rolling{w}_mean_{col}
        - rolling{w}_max_{col}
        - rolling{w}_min_{col}
        - rolling_next_{w}_mean_{col}
        - rolling_next_{w}_max_{col}
        - rolling_next_{w}_min_{col}
    """
    s = _as_float32(df[col])
    new_cols = {}

    for wndw in windows:
        r = s.rolling(window=wndw, min_periods=wndw)
        sf = s.shift(-wndw)
        rf = sf.rolling(window=wndw, min_periods=wndw)

        new_cols[f"rolling{wndw}_mean_{col}"] = _as_float32(r.mean())
        new_cols[f"rolling{wndw}_max_{col}"] = _as_float32(r.max())
        new_cols[f"rolling{wndw}_min_{col}"] = _as_float32(r.min())

        new_cols[f"rolling_next_{wndw}_mean_{col}"] = _as_float32(rf.mean())
        new_cols[f"rolling_next_{wndw}_max_{col}"] = _as_float32(rf.max())
        new_cols[f"rolling_next_{wndw}_min_{col}"] = _as_float32(rf.min())

    return _concat_new_cols(df, new_cols)


def add_StatisticsDifferenceFeatures(df, col, stat="daily"):
    """
    Feature extraction function for the final feature set.
    
    The input column `col` is typically "X" (total energy use).
    For total energy use X (or another column passed as `col`), this function adds
    features that describe how the value of X at each time step deviates from daily
    or yearly values. For example, when called with stat="daily", the function
    subtracts the daily mean, maximum, minimum, standard deviation, and variance of X
    from the current value of X at each time step. When called with stat="yearly",
    the same operations are performed using yearly statistics instead. The reference
    values must already be present in the dataset.
    
    Parameters
    ----------
    df : pandas.DataFrame
        Final features dataset containing column X (total energy use or imported
        electricity) and the corresponding daily or yearly statistic columns.
    col : str
        Name of the column to use. In practice this is typically "X".
    stat : str, optional
        Statistic time scale to use. Typical values are "daily" and "yearly".
        Default is "daily".
    
    Returns
    -------
    pandas.DataFrame
        Final features dataset with additional features added at every time step:
        - X - {stat}_mean_X
        - X - {stat}_max_X
        - X - {stat}_min_X
        - X - {stat}_std_X
        - X - {stat}_var_X
        - {stat}_mean_X_vs_{stat}_max_X
        - {stat}_mean_X_vs_{stat}_std_X
        - {stat}_mean_X_vs_{stat}_var_X
    """
    x = _as_float32(df[col])
    mean_c = _as_float32(df[f"{stat}_mean_{col}"])
    max_c = _as_float32(df[f"{stat}_max_{col}"])
    min_c = _as_float32(df[f"{stat}_min_{col}"])
    std_c = _as_float32(df[f"{stat}_std_{col}"])
    var_c = _as_float32(df[f"{stat}_var_{col}"])

    new_cols = {
        f"{col} - {stat}_mean_{col}": _as_float32(x - mean_c),
        f"{col} - {stat}_max_{col}": _as_float32(x - max_c),
        f"{col} - {stat}_min_{col}": _as_float32(x - min_c),
        f"{col} - {stat}_std_{col}": _as_float32(x - std_c),
        f"{col} - {stat}_var_{col}": _as_float32(x - var_c),
    }

    with np.errstate(divide="ignore", invalid="ignore"):
        new_cols[f"{stat}_mean_{col}_vs_{stat}_max_{col}"] = _as_float32(mean_c / max_c)
        new_cols[f"{stat}_mean_{col}_vs_{stat}_std_{col}"] = _as_float32(mean_c / std_c)
        new_cols[f"{stat}_mean_{col}_vs_{stat}_var_{col}"] = _as_float32(mean_c / var_c)

    return _concat_new_cols(df, new_cols)


def add_LagFeatures(df, col, lag):
    """
    Feature extraction function for the final feature set.

    For the input column `col` (typically "X", total energy use), this function
    adds lagged and lead values and their differences to the current value. For
    each time step and for each i in [1, ..., lag], the value of `col` i hours
    earlier and i hours later is added, together with the corresponding
    differences relative to the current value.

    Parameters
    ----------
    df : pandas.DataFrame
        Final features dataset containing column `col`.
    col : str
        Name of the column to use.
    lag : int
        Maximum lag and lead, given in number of hours.

    Returns
    -------
    pandas.DataFrame
        Final features dataset with additional features added at every time step.
        For each i in [1, ..., lag], the following columns are added:
        - {col}-{i}h
        - {col}_diff_{i}h
        - {col}+{i}h
        - {col}_diff_+{i}h
    """
    s = _as_float32(df[col])
    new_cols = {}

    for i in range(1, lag + 1):
        past = s.shift(i)
        fut = s.shift(-i)

        new_cols[f"{col}-{i}h"] = _as_float32(past)
        new_cols[f"{col}_diff_{i}h"] = _as_float32(s - past)

        new_cols[f"{col}+{i}h"] = _as_float32(fut)
        new_cols[f"{col}_diff_+{i}h"] = _as_float32(s - fut)

    return _concat_new_cols(df, new_cols)


def add_corrFeatures(df, column1, column2, time_step):
    """
     Feature extraction function for the final feature set.
    
     For two input columns (e.g. "X" for total energy use and "Tout" for outdoor
     temperature), this function computes Pearson and Spearman correlation
     coefficients over blocks of length `time_step` (in hours) and assigns the
     same correlation value to all time steps within each block.
    
     Parameters
     ----------
     df : pandas.DataFrame
         Final features dataset containing columns `column1` and `column2`.
     column1 : str
         Name of the first column (e.g. "X", total energy use).
     column2 : str
         Name of the second column (e.g. "Tout", outdoor temperature).
     time_step : int
         Block length in number of rows (hours), e.g. 24 or 168.
    
     Returns
     -------
     pandas.DataFrame
         Final features dataset with additional correlation features added at
         every time step:
         - pearson_{column1}_{column2}_{time_step}
         - spearman_{column1}_{column2}_{time_step}
     """
    if column1 == column2:
        return df

    n = len(df)
    pearson_corr = np.full(n, np.nan, dtype=np.float32)
    spearman_corr = np.full(n, np.nan, dtype=np.float32)

    a = df[column1].to_numpy(dtype=float, copy=False)
    b = df[column2].to_numpy(dtype=float, copy=False)

    with warnings.catch_warnings():
        warnings.simplefilter("ignore", ConstantInputWarning)

        for start in range(0, n, time_step):
            end = min(start + time_step, n)

            x = a[start:end]
            y = b[start:end]
            mask = ~np.isnan(x) & ~np.isnan(y)

            if mask.sum() < 2:
                pearson_val = np.nan
                spearman_val = np.nan
            else:
                pearson_val = stats.pearsonr(x[mask], y[mask])[0]
                spearman_val = stats.spearmanr(x[mask], y[mask])[0]

            pearson_corr[start:end] = pearson_val
            spearman_corr[start:end] = spearman_val

    df[f"pearson_{column1}_{column2}_{time_step}"] = pearson_corr
    df[f"spearman_{column1}_{column2}_{time_step}"] = spearman_corr
    return df


def add_linregFeatures(df, column1, column2):
    """
    Feature extraction function for the final feature set.

    For two input columns (e.g. "X" for total energy use and "Tout" for outdoor
    temperature), this function fits a linear regression model predicting
    `column2` from `column1` using rows where both columns are available. The
    function then adds the predicted values at each time step (where `column1`
    is available) and appends model performance metrics and fitted parameters
    as additional features.

    Parameters
    ----------
    df : pandas.DataFrame
        Final features dataset containing columns `column1` and `column2`.
    column1 : str
        Name of the predictor column (e.g. "X", total energy use).
    column2 : str
        Name of the target column (e.g. "Tout", outdoor temperature).

    Returns
    -------
    pandas.DataFrame
        Final features dataset with additional linear-regression feature columns:
        - linreg_{column2}_from_{column1}_pred
        - linreg_{column2}_from_{column1}_R2
        - linreg_{column2}_from_{column1}_RMSE
        - linreg_{column2}_from_{column1}_MAE
        - linreg_{column2}_from_{column1}_CV
        - linreg_{column2}_from_{column1}_slope
        - linreg_{column2}_from_{column1}_intercept
    """
    if column1 == column2:
        return df

    mask = df[[column1, column2]].notna().all(axis=1)
    if mask.sum() < 2:
        return df

    x = df.loc[mask, column1].to_numpy(dtype=float).reshape(-1, 1)
    y = df.loc[mask, column2].to_numpy(dtype=float).reshape(-1, 1)

    model = LinearRegression()
    model.fit(x, y)

    preds = np.full(len(df), np.nan, dtype=np.float32)
    valid_x_mask = df[column1].notna()
    preds[valid_x_mask.to_numpy()] = model.predict(
        df.loc[valid_x_mask, column1].to_numpy(dtype=float).reshape(-1, 1)
    ).ravel().astype(np.float32)

    base = f"linreg_{column2}_from_{column1}"
    df[f"{base}_pred"] = preds

    y_fit = y.ravel()
    y_pred_fit = model.predict(x).ravel()

    r2 = float(r2_score(y_fit, y_pred_fit))
    rmse = float(np.sqrt(mean_squared_error(y_fit, y_pred_fit)))
    mae = float(mean_absolute_error(y_fit, y_pred_fit))
    mean_y = float(np.mean(y_fit))
    cv = float(rmse / mean_y) if not np.isclose(mean_y, 0) else np.nan

    slope = float(model.coef_[0][0])
    intercept = float(model.intercept_[0])

    df[f"{base}_R2"] = np.float32(r2)
    df[f"{base}_RMSE"] = np.float32(rmse)
    df[f"{base}_MAE"] = np.float32(mae)
    df[f"{base}_CV"] = np.float32(cv)
    df[f"{base}_slope"] = np.float32(slope)
    df[f"{base}_intercept"] = np.float32(intercept)

    return df


def add_autocorr_global(df, column, lags):
    """
    Feature extraction function for the final feature set.

    For a single input column (e.g. "X", total energy use), this function computes
    Pearson autocorrelation coefficients for multiple lags using the entire time
    series. For each lag, one autocorrelation value is computed and added as a
    constant feature across all time steps.

    Parameters
    ----------
    df : pandas.DataFrame
        Final features dataset containing column `column`.
    column : str
        Name of the column to use (e.g. "X", total energy use).
    lags : iterable
        Lags in number of time steps (hours), e.g. [1, 2, 3, 24, 168].

    Returns
    -------
    pandas.DataFrame
        Final features dataset with additional autocorrelation features:
        - autocorr_{column}_lag{lag}
    """
    x = df[column].to_numpy(dtype=float, copy=False)

    for lag in lags:
        x1 = x[lag:]
        x2 = x[:-lag]
        mask = ~np.isnan(x1) & ~np.isnan(x2)

        if mask.sum() < 2:
            corr = np.nan
        else:
            corr = stats.pearsonr(x1[mask], x2[mask])[0]

        df[f"autocorr_{column}_lag{lag}"] = np.float32(corr) if corr == corr else np.nan

    return df


def add_autocorr_blockwise(df, column, lag, time_step):
    """
    Feature extraction function for the final feature set.

    Computes the Pearson autocorrelation of a time series column (e.g. "X" for
    total energy use) at a given lag (e.g. lag=1 for consecutive hourly values or
    lag=24 for values one day apart) within consecutive, non-overlapping blocks
    of length `time_step` (e.g. time_step=24 for daily blocks or time_step=168
    for weekly blocks). The same autocorrelation value is assigned to all rows
    within each block.

    Parameters
    ----------
    df : pandas.DataFrame
        Input DataFrame containing the time series.
    column : str
        Name of the column to compute autocorrelation for, typically "X"
        representing total energy use.
    lag : int
        Lag (number of time steps) used for autocorrelation.
    time_step : int
        Block size over which autocorrelation is computed.

    Returns
    -------
    pandas.DataFrame
        Final features dataset with additionanal autocorrelation feature column:
        - `autocorr_<column>_lag<lag>_step<time_step>`.
    """
    n = len(df)
    autocorr_vals = np.full(n, np.nan, dtype=np.float32)

    x = df[column].to_numpy(dtype=float, copy=False)

    for start in range(0, n, time_step):
        end = min(start + time_step, n)
        subset = x[start:end]

        if len(subset) <= lag:
            corr = np.nan
        else:
            x1 = subset[lag:]
            x2 = subset[:-lag]
            if np.nanstd(x1) == 0 or np.nanstd(x2) == 0:
                corr = np.nan
            else:
                corr = stats.pearsonr(x1, x2)[0]

        autocorr_vals[start:end] = corr

    df[f"autocorr_{column}_lag{lag}_step{time_step}"] = autocorr_vals
    return df


def _sax_day_probability_series(df, col, symbol_count, breakfreq):
    """
    Feature extraction helper for SAX-based features.

    Computes the probability of each day's time-series pattern using
    Symbolic Aggregate approXimation (SAX), which represents a normalized
    time series as a sequence of discrete symbols describing its shape.
    Each day is encoded as a string of symbols, and the returned value
    reflects how frequently that daily pattern occurs in the data.

    The input column is typically "X" (total energy use), and an example SAX
    configuration is `symbol_count=6` with `breakfreq="4h"`, which encodes
    the daily load shape using 6 symbols at 4-hour resolution.

    Parameters
    ----------
    df : pandas.DataFrame
        Input DataFrame containing a TimeStamp column and the signal column.
    col : str
        Name of the signal column, typically "X" representing total energy use.
    symbol_count : int
        Number of discrete SAX symbols used to encode the 'signal'
        (e.g. 6 symbols).
    breakfreq : str
        Resampling frequency before SAX encoding
        (e.g. "4h" for 4-hour averages).

    Returns
    -------
    pandas.Series
        Series indexed by date, where each value is the probability of
        observing that day's SAX-encoded pattern.
    """
    sax_df = df[["TimeStamp", col]].set_index("TimeStamp")
    sax_df = sax_df.ffill().dropna()
    sax_df = (sax_df - sax_df.mean()) / sax_df.std()

    breakpoints = norm.ppf(np.linspace(1.0 / symbol_count, 1 - 1.0 / symbol_count, symbol_count - 1))
    breakpoints = np.concatenate((breakpoints, np.array([np.inf])))

    sax_df = sax_df.resample(breakfreq).mean().dropna()
    vals = sax_df[col].to_numpy(dtype=float, copy=False)

    steps = np.searchsorted(breakpoints, vals, side="left").astype(np.int16)
    letters = np.array([string.ascii_letters[int(i)] for i in steps], dtype=object)

    idx = sax_df.index
    tmp = pd.DataFrame({"Date": idx.date, "Time": idx.time, "letter": letters})
    pivot = tmp.pivot(index="Date", columns="Time", values="letter")

    day_string = pivot.dropna().sum(axis=1)
    probs = day_string.value_counts() / day_string.count()

    return day_string.map(probs).astype(np.float32)


def add_SAXFeatures_many(df, col, sax_set):
    """
    Feature extraction function for the final feature set.

    Adds SAX-based daily pattern probability features by calling
    `_sax_day_probability_series` for each `(symbol_count, breakfreq)` pair in
    `sax_set`. Each feature represents how likely a given day's load pattern is
    relative to other days in the same building.

    The signal column is typically "X" (total energy use). An example `sax_set`
    may include combinations such as `(6, "4h")`, `(8, "12h")`, or `(12, "24h")`,
    corresponding to different symbol resolutions and time aggregations.

    Parameters
    ----------
    df : pandas.DataFrame
        Input DataFrame containing a TimeStamp column and the signal column.
    col : str
        Name of the signal column, typically "X" representing total energy use.
    sax_set : iterable
        Iterable of `(symbol_count, breakfreq)` pairs passed to
        `_sax_day_probability_series`.

    Returns
    -------
      pandas.DataFrame
            Final features dataset with additional SAX-based feature columns:
            - SAX_<col>_<symbol_count>_<breakfreq>  (one column per entry in `sax_set`)

    """
    date_key = df["TimeStamp"].dt.date
    new_cols = {}

    for symbol_count, breakfreq in sax_set:
        feature_name = f"SAX_{col}_{symbol_count}_{breakfreq}"
        day_prob = _sax_day_probability_series(df, col, symbol_count, breakfreq)
        new_cols[feature_name] = _as_float32(date_key.map(day_prob))

    return _concat_new_cols(df, new_cols)


def add_rolling_drop_features(df, col, wndw=3, shft=5):
    """
    Feature extraction function for the final feature set.

    Adds a rolling drop feature computed as the difference between a rolling
    mean and a time-shifted version of the same rolling mean:
    `rolling_mean(wndw) - rolling_mean(wndw).shift(shft)`. This captures
    changes in the average level of the signal over time.

    The signal column is typically "X" (total energy use). For example,
    `wndw=3` and `shft=5` compare the current 3-step rolling mean to the value
    five time steps earlier.

    Parameters
    ----------
    df : pandas.DataFrame
        Input DataFrame containing the signal.
    col : str
        Name of the signal column, typically "X" representing total energy use.
    wndw : int, optional
        Window size used to compute the rolling mean.
    shft : int, optional
        Number of time steps by which the rolling mean is shifted.

    Returns
    -------
    pandas.DataFrame
        Final features dataset with the additional rolling drop feature column:
        - rolling_drop_<wndw>_<shft>
    """
    s = _as_float32(df[col])
    cur = s.rolling(window=wndw, min_periods=wndw).mean()
    drop_feature_name = f"rolling_drop_{wndw}_{shft}"
    df[drop_feature_name] = _as_float32(cur - cur.shift(shft))
    return df


def add_classification_features(df, df_orig, col):
    """
    Feature extraction function for the final feature set.
    
    Add building classification probability features to a feature DataFrame.
    This function appends constant building type classification probabilities
    as features. The probabilities are obtained by applying the
    CofClassification model to a building-level electricity time series and
    represent the likelihood of the building belonging to each building type
    class. Each class combines a building category with its dominant heating
    type (electric or non-electric).

    The CofClassification model is designed to classify total electricity use
    ("ElImp"). In some cases, classification must be performed on a different
    signal (e.g. x representing total energy use). In such cases, the selected
    signal is temporarily treated as total electricity import and mapped to
    the "ElImp" column prior to classification.

    The predicted probabilities are added as constant columns, one per building
    class, and are identical for all time steps corresponding to the same
    building.

    Parameters
    ----------
    df : pandas.DataFrame
        Feature DataFrame to which classification probability features are added.
    df_orig : pandas.DataFrame
        Original building time series used as input to the building
        classification model.
    col : str
        Name of the column in df_orig containing the signal to be classified
        (typically x). If col is not "ElImp", the signal is copied to the
        "ElImp" column before classification.

    Returns
    -------
    pandas.DataFrame
        Feature DataFrame with additional classification probability columns:
        - bcat_class_probability_<building_class>
    """
    if col != "ElImp":
        df_orig = df_orig.copy()
        df_orig["ElImp"] = df_orig[col]

    class_prob = CofClassifyClassify.predict_class_probabilities_ts(df_orig)

    for cls in class_prob.columns:
        df[f"bcat_class_probability_{cls}"] = np.float32(class_prob[cls].iloc[0])

    return df

def add_ET_curve_features(ts, df,col="X"):
    """
    Feature extraction function for the final feature set.
    
    Adds ET-curve features based on a 2-segment piecewise linear fit of the signal
    column (`col`, typically "X" for total energy use) as a function of outdoor
    temperature ("Tout"). The change point temperature (CPT) and the slopes below
    and above the CPT (B1 and B2) are extracted from the fit, together with B0 (the
    fitted value at the CPT from the below-CPT regression). The extracted values
    are added as constant features and repeated for all time steps.
    
    Parameters
    ----------
    ts : pandas.DataFrame
        Final features dataset that receives the ET-curve feature columns.
    df : pandas.DataFrame
        Original building time series containing "Tout" and the signal column `col`.
    col : str, optional
        Name of the signal column to use in the ET-curve fit, typically "X"
        representing total energy use.
    
    Returns
    -------
    pandas.DataFrame
        Final features dataset with additional ET-curve feature columns:
        - EH_features_h_CPT
        - EH_features_h_B0
        - EH_features_h_B1
        - EH_features_h_B2

    """
    CPT = B0 = B1 = B2 = np.nan

    try:
        mask = (~df[col].isna()) & (~df["Tout"].isna())
        T = df.loc[mask, "Tout"].to_numpy(dtype=float)
        E = df.loc[mask, col].to_numpy(dtype=float)

        if T.size < 4:
            raise ValueError("Not enough points for a 2-segment fit")

        model = pwlf.PiecewiseLinFit(T, E)
        breaks = model.fit(2)              # [min(T), CPT, max(T)]
        CPT = float(breaks[1])

        below_mask = T <= CPT
        above_mask = T > CPT
        if below_mask.sum() < 2 or above_mask.sum() < 2:
            raise ValueError("Not enough points on one side of CPT")

        # Below CPT
        reg_below = LinearRegression().fit(T[below_mask].reshape(-1, 1), E[below_mask])
        B1 = float(reg_below.coef_[0])
        B0 = float(reg_below.predict(np.array([[CPT]], dtype=float))[0])

        # Above CPT
        reg_above = LinearRegression().fit(T[above_mask].reshape(-1, 1), E[above_mask])
        B2 = float(reg_above.coef_[0])

    except Exception:
        pass

    ts["EH_features_h_CPT"] = np.float32(CPT) if np.isfinite(CPT) else np.nan
    ts["EH_features_h_B0"] = np.float32(B0) if np.isfinite(B0) else np.nan
    ts["EH_features_h_B1"] = np.float32(B1) if np.isfinite(B1) else np.nan
    ts["EH_features_h_B2"] = np.float32(B2) if np.isfinite(B2) else np.nan
    return ts


def get_extracted_features_from_building_ts(ts, building_id="building", savedir=None, forced_X=None):
    """
    Extract the disaggregation feature matrix from a building time series.
    
    This function extracts features for every time step in a building
    time series, using the feature extraction functions. 
    It returns a feature extracted time series for the building. 
    Each row in the returned DataFrame corresponds to one time step of the original building file.
    
    The input time series must contain an outdoor temperature series ("Tout"), timestamps
    ("TimeStamp"), and a total imported energy use column used as the main signal "X".
    If "X" is not provided/specified, it is derived from the total electricity use column. 
    Depending on the data source, electricity use may be provided either as
    imported electricity ("ElImp") or as total electricity use ("ElTot").
    If both are present, the column with the largest total energy is selected
    as "X". The selected signal is used consistently for feature extraction.
    
    In addition to time- and statistics-based features, the function appends
    building type classification probabilities as constant features.
    The classification model is designed to classify imported electricity
    ("ElImp"). When the main signal is not "ElImp" (e.g. "X" represents total
    electricity use from "ElTot"), that signal is temporarily mapped to
    "ElImp" for the purpose of classification.
    
    Parameters
    ----------
    ts : pandas.DataFrame
        Building time series. Must contain:
        - "TimeStamp"
        - "Tout"
        - and either "X" or one of {"ElImp", "ElTot"} (or the column specified
          by forced_X).
    building_id : str, optional
        Building identifier (building name or ID) used for logging and optional saving.
    savedir : str, optional
        If provided, save extracted features to this folder as a .txt file
        using ';' as separator.
    forced_X : str, optional
        Column name in ts to force as the main signal "X".
    
    Returns
    -------
    pandas.DataFrame
        Feature DataFrame for a building to be used for the disaggregation model, containing one row per
        time step and all engineered features, including constant classification
        probability columns.
        The "TimeStamp" column is removed in the returned DataFrame.
    """

    if building_id is None:
        building_id = "building"
    print(f"Extracting disaggregation features for {building_id}.")

    if forced_X is None:
        X_cols = [c for c in ["ElImp", "ElTot"] if c in ts.columns]
        if "X" not in ts.columns and X_cols:
            biggest = ts[X_cols].sum().idxmax()
            ts["X"] = ts[biggest]
    else:
        if forced_X not in ts.columns:
            raise ValueError(f"forced_X='{forced_X}' not found in ts columns")
        ts["X"] = ts[forced_X]

    sax_set = [
        [6, "4h"], [6, "6h"], [6, "12h"], [6, "24h"],
        [8, "4h"], [8, "6h"], [8, "12h"], [8, "24h"],
        [12, "4h"], [12, "6h"], [12, "12h"], [12, "24h"],
    ]

    ts_orig = ts.copy()

    ts = ts.reset_index(drop=False)
    ts = ts[["Tout", "X", "TimeStamp"]]
    ts["Tout"] = _as_float32(ts["Tout"])
    ts["X"] = _as_float32(ts["X"])

    ts = add_date_features(ts)
    ts = add_dailyStatisticsFeatures(ts, "X")
    ts = add_yearlyStatisticsFeatures(ts, "X")
    ts = add_rollingStatisticsFeatures_multi(ts, "X", windows=range(1, 25))
    ts = add_rolling_drop_features(ts, "X", 3, 5)
    ts = add_rolling_drop_features(ts, "X", 3, 4)
    ts = add_rolling_drop_features(ts, "X", 2, 5)
    ts = add_rolling_drop_features(ts, "X", 2, 4)
    ts = add_rolling_drop_features(ts, "X", 2, 3)
    ts = add_StatisticsDifferenceFeatures(ts, "X", "daily")
    ts = add_StatisticsDifferenceFeatures(ts, "X", "yearly")
    ts = add_LagFeatures(ts, "X", 12)
    ts = add_linregFeatures(ts, "X", "Tout")
    ts = add_corrFeatures(ts, "X", "Tout", 24)
    ts = add_corrFeatures(ts, "X", "Tout", 168)
    ts = add_autocorr_global(ts, "X", lags=[1, 2, 3, 24, 168])
    ts = add_autocorr_blockwise(ts, "X", 1, 24)
    ts = add_autocorr_blockwise(ts, "X", 2, 24)
    ts = add_autocorr_blockwise(ts, "X", 24, 168)
    ts = add_ET_curve_features(ts, ts, "X")
    ts = add_SAXFeatures_many(ts, "X", sax_set)
    ts = add_classification_features(ts, ts_orig, "X")
    
    ts = ts.copy()
    ts = ts.drop(columns=["TimeStamp"])

    print("Disaggregation features extracted.")

    if savedir:
        os.makedirs(savedir, exist_ok=True)
        ts.to_csv(os.path.join(savedir, building_id + ".txt"), sep=";")
        print(f"Saved to {savedir}")

    return ts


def load_ts_data(path, ext="txt"):
    """
    Helper function.
    Load time series data from a file or a folder.
    
    If `path` points to a single file, the file is read using
    `treaData.getTreaCsv`. If `path` points to a directory, all matching files
    in the folder are read using `treaData.readAllcsvsFromFolder`.
    
    Parameters
    ----------
    path : str
        Path to a file or a directory containing time series data.
    ext : str, optional
        File extension to read when `path` is a directory (default "txt").
    
    Returns
    -------
    tuple
        A tuple (mm, dd) as returned by the underlying treaData reader functions.
    """
    if os.path.isfile(path):
        mm,dd=treaData.getTreaCsv(path)

    if os.path.isdir(path):
        mm,dd=treaData.readAllcsvsFromFolder(path, ext)
    return(mm,dd)


def get_extracted_features_from_file(path, savedir=None):
    """
    Extract disaggregation features from a single building file.

    Parameters
    ----------
    path : str
        Building file path.
    savedir : str, optional
        Output folder for extracted features.

    Returns
    -------
    ts : pandas.DataFrame
        Extracted features dataframe.
    """
    building_name = os.path.basename(path)
    building_name = building_name.replace(".txt", "").replace(".csv", "")
    meta, ts = load_ts_data(path)
    building_features = get_extracted_features_from_building_ts(ts, building_name, savedir)
    return building_features


def get_extracted_features_from_building_dict(ts_dict, savedir=None):
    """
    Extract disaggregation features for multiple buildings stored as time series in a dict.

    Parameters
    ----------
    ts_dict : dict
        Dict of {building_id: time series df}.
    savedir : str, optional
        Output folder.

    Returns
    -------
    d : dict
        Dict of {building_id: extracted features df}.
    """
    feature_extraxted_ts_dict = {}
    for key in ts_dict.keys():
        feature_extraxted_ts_dict[key] = get_extracted_features_from_building_ts(ts_dict[key], key, savedir)
    return feature_extraxted_ts_dict