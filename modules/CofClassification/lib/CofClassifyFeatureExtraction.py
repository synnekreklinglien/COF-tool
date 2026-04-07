# -*- coding: utf-8 -*-
# -----------------------------------------------------------------------------
# Author: Synne Krekling Lien
# Contact: synnekreklinglien@gmail.com
# Date: 24.02.2026
# Repository: https://github.com/synnekreklinglien/COF-tool
#
# Feature extraction from building time series data for
# supervised classification of building category and heating type
# using hourly electricity smart meter measurements and outdoor temperature.
# The classification features extracts one set of features from the full
# time series of a building, a total of 180 features and one row
# for any length time series. 
# -----------------------------------------------------------------------------


import copy
import os

import holidays
import numpy as np
import pandas as pd
import pwlf
from sklearn.linear_model import LinearRegression

from . import ReadtreaData as treaData
print("Imported COF-Tool CofClassifyFeatureExtraction.")

#Import files needed for feature extraction
base_dir = os.path.dirname(__file__) if "__file__" in globals() else os.getcwd()
reaDir = os.path.join(base_dir, "../resources/")
heating_categories = pd.read_csv(reaDir+'heat_source.csv', sep = ";",index_col = "heat_source")



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


def load_daily_standard_load_profiles(path):
    """
    Get background data needed for feature generation, needed for the function 
    "add_daily_standard_load_profiles_correlation_features".
    Standard daily load profiles (SLPs) are mean daily load profiles for different 
    building categories developed with the tool Building-PROFet version 2. 
    This function is a helper function used to load the 
    standard daily load profiles for the full year and winter from the resources
    where they are stored as CSV files.
    The function reads all CSV files in the subfolders `slp_daily/` and `slp_daily_winter/`
    into pandas DataFrames and stores them for the upcoming feature extraction. 

    Parameters
    ----------
    path : str
        Base directory containing 'slp_daily/' and 'slp_daily_winter/' folders.

    Returns
    -------
    tuple of dict
        (daily_slp, daily_slp_winter)
        - daily_slp: dict of DataFrames from 'slp_daily/'
        - daily_slp_winter: dict of DataFrames from 'slp_daily_winter/'
    """
    daily_slp = {}
    daily_slp_winter = {}

    for file in os.listdir(os.path.join(path, "slp_daily")):
        file_path = os.path.join(path, "slp_daily", file)
        df = pd.read_csv(file_path)
        file_name = os.path.splitext(file)[0]
        daily_slp[file_name] = df

    for file in os.listdir(os.path.join(path, "slp_daily_winter")):
        file_path = os.path.join(path, "slp_daily_winter", file)
        df = pd.read_csv(file_path)
        file_name = os.path.splitext(file)[0]
        daily_slp_winter[file_name] = df

    return daily_slp, daily_slp_winter


#Load the standard daily load profiles before feature extraction.
daily_slp, daily_slp_winter = load_daily_standard_load_profiles(reaDir)


def df_normalizer(df, cols):
    """
    Helper function needed for the function "get_extracted_features_from_building_ts". 
    Normalize selected columns of a DataFrame to the [0, 1] range.

    Each specified column is independently scaled using min-max normalization
    based on the minimum and maximum values in the input DataFrame. The function
    is typically used to normalize the imported electricity column ("ElImp"),
    but can be applied to any numeric columns.

    Parameters
    ----------
    df : pandas.DataFrame
        Input DataFrame containing building time series data.
    cols : list of str
        Names of the columns to normalize.

    Returns
    -------
    pandas.DataFrame
        A deep-copied DataFrame with the specified columns normalized.
    """

    df_normalized = copy.deepcopy(df)

    for col in cols:
        x_min = df[col].min()
        x_max = df[col].max()

        df_normalized[col] = (df_normalized[col] - x_min) / (x_max - x_min)

    return df_normalized


def add_holiday_feature(df, holiday_weeks=[9, 26, 27, 28, 29, 30, 31, 32, 33, 40, 51, 52, 53]):
    """
    Add a binary holiday indicator based on Norwegian public holidays.

    This helper function is used by `add_date_features` to create an
    `IsHoliday` column. Dates are marked as holidays if they coincide with
    official Norwegian public holidays or fall within user-specified
    holiday weeks.

    Parameters
    ----------
    df : pandas.DataFrame
        Input DataFrame containing 'TimeStamp' (datetime) and 'weekofyear' columns.
    holiday_weeks : list of int, optional
        Week numbers that should additionally be treated as holidays
        (default: [9, 26, 27, 28, 29, 30, 31, 32, 33, 40, 51, 52, 53]).

    Returns
    -------
    pandas.DataFrame
        Copy of the input DataFrame with an added 'IsHoliday' column
        (1 = holiday, 0 = non-holiday).
    """
    df = df.copy()
    NO_holidays = holidays.NO()

    def is_norwegian_holiday(date):
        return 1 if date in NO_holidays else 0

    df["IsHoliday"] = df["TimeStamp"].dt.date.apply(is_norwegian_holiday)
    df.loc[(df["weekofyear"].isin(holiday_weeks)) & (df["IsHoliday"] == 0), "IsHoliday"] = 1
    return df


def add_date_features(df, holiday_weeks=[9, 26, 27, 28, 29, 30, 31, 32, 33, 40, 51, 52, 53]):
    """
    Add temporal helper features derived from the TimeStamp column.

    This function creates time-, calendar-, and season-related features that are
    used internally by `get_extracted_features_from_building_ts`. The generated
    features are not intended for direct use in the final feature set, but serve
    as intermediate variables for extracting higher-level features.

    Features are derived from the 'TimeStamp' datetime column and include standard
    calendar fields, binary indicators, and trigonometric encodings for
    time representation. A binary holiday indicator is added via
    `add_holiday_feature()`.

    Parameters
    ----------
    df : pandas.DataFrame
        Input DataFrame containing a 'TimeStamp' column of dtype datetime.
    holiday_weeks : list of int, optional
         week numbers that should be additionally treated as holidays
        (default: [9, 26, 27, 28, 29, 30, 31, 32, 33, 40, 51, 52, 53]).

    Returns
    -------
    pandas.DataFrame
        Copy of the input DataFrame with the following added columns:
        hour, day, month, dayofyear, weekofyear, dayofweek, season, weekend,
        IsHoliday, day_night, sin_hour, cos_hour,
        sin_day_weekendness, cos_day_unweekendness.
    """
    df = df.copy()
    seasons = {1: 1, 2: 1, 3: 2, 4: 2, 5: 2, 6: 3, 7: 3, 8: 3, 9: 4, 10: 4, 11: 4, 12: 1}

    df["hour"] = df.TimeStamp.dt.hour
    df["day"] = df.TimeStamp.dt.day
    df["month"] = df.TimeStamp.dt.month
    df["dayofyear"] = df.TimeStamp.dt.day_of_year
    df["weekofyear"] = (df["dayofyear"] // 7) + 1
    df["dayofweek"] = df.TimeStamp.dt.dayofweek
    df["season"] = df["TimeStamp"].dt.month.map(seasons)
    df["weekend"] = np.where(df["dayofweek"] > 4, 1, 0)
    df = add_holiday_feature(df, holiday_weeks)
    df["day_night"] = df["hour"].apply(lambda x: 1 if 8 <= x <= 20 else 0)
    df["sin_hour"] = np.sin(2 * np.pi * df["hour"] / 24)
    df["cos_hour"] = np.cos(2 * np.pi * df["hour"] / 24)
    df["sin_day_weekendness"] = np.sin(0.5 * np.pi * (df["dayofweek"] + 1) / 7)
    df["cos_day_unweekendness"] = np.cos(0.5 * np.pi * (df["dayofweek"] + 1) / 7)

    return df


def add_min_max_mean_features(d, df, target, filtered=None, filter_val=None):
    """
    Feature extraction function for the final feature set.

    Computes the max, and mean of a target column (typically "ElImp"),
    the max/mean ratio, and the hour of day when the maximum value occurs.

    If `filtered` and `filter_val` are provided, the features are computed only
    for rows where `df[filtered] == filter_val`. For example, if `filtered` is
    "season" and `filter_val` is 1, the features are extracted only for winter
    months.

    Parameters
    ----------
    d : dict
        Dictionary to update with the computed features.
    df : pandas.DataFrame
        DataFrame containing the target column and optionally a filtering column.
    target : str
        Column name to extract features from.
    filtered : str, optional
        Column name used for filtering (default None).
    filter_val : any, optional
        Value used for filtering (default None).

    Returns
    -------
    dict
        Updated dictionary containing extracted features named:
        - {target}_max_{filter_info}
        - {target}_mean_{filter_info}
        - {target}_max_vs_mean_{filter_info}
        - {target}_max_hour_{filter_info}
    """
    if filtered is not None and filter_val is not None:
        filtered_df = df[df[filtered] == filter_val]
    else:
        filtered_df = df

    max_value = filtered_df[target].max()
    mean_value = filtered_df[target].mean()
    max_vs_mean_value = max_value / mean_value if mean_value else np.nan


    if filtered is not None and filter_val is not None:
        filter_info = f"_{filtered}_is_{str(filter_val)}"
    else:
        filter_info = ""

    d[f"{target}_max{filter_info}"] = max_value
    d[f"{target}_mean{filter_info}"] = mean_value
    d[f"{target}_max_vs_mean{filter_info}"] = max_vs_mean_value

    if "hour" in filtered_df.columns and not filtered_df[filtered_df[target] == max_value].empty:
        max_hour = filtered_df.loc[filtered_df[target] == max_value, "hour"].values[0]
        d[f"{target}_max_hour{filter_info}"] = max_hour
    else:
        d[f"{target}_max_hour{filter_info}"] = None

    return d


def add_daytime_features(d: dict, df: pd.DataFrame) -> dict:
    """
    Feature extraction function for the final feature set.

    Computes time-of-day energy consumption features based on "ElImp" by
    splitting the day into night (17:00–06:00), morning (06:00–12:00),
    and day (12:00–17:00). For each period, the mean, standard deviation,
    minimum, and maximum values are computed.

    Parameters
    ----------
    d : dict
        Dictionary to update with the computed features.
    df : pandas.DataFrame
        DataFrame containing the columns "ElImp" and "hour" (0–23).

    Returns
    -------
    dict
        Updated dictionary containing extracted features named:
        - night_ElImp_mean, night_ElImp_std, night_ElImp_min, night_ElImp_max
        - morning_ElImp_mean, morning_ElImp_std, morning_ElImp_min, morning_ElImp_max
        - day_ElImp_mean, day_ElImp_std, day_ElImp_min, day_ElImp_max
    """
    if "ElImp" not in df or "hour" not in df:
        for p in ["night", "morning", "day"]:
            for stat in ["mean", "std", "min", "max"]:
                d[f"{p}_ElImp_{stat}"] = np.nan
        return d

    night_filter = (df["hour"] >= 17) | (df["hour"] < 6)
    morning_filter = (df["hour"] >= 6) & (df["hour"] < 12)
    day_filter = (df["hour"] >= 12) & (df["hour"] < 17)

    periods = {"night": night_filter, "morning": morning_filter, "day": day_filter}
    for period, mask in periods.items():
        subset = df.loc[mask, "ElImp"]
        d[f"{period}_ElImp_mean"] = float(subset.mean()) if not subset.empty else np.nan
        d[f"{period}_ElImp_std"] = float(subset.std()) if not subset.empty else np.nan
        d[f"{period}_ElImp_min"] = float(subset.min()) if not subset.empty else np.nan
        d[f"{period}_ElImp_max"] = float(subset.max()) if not subset.empty else np.nan

    return d


def add_min_max_mean_below_feature(d, df, target, filtered=None, filter_val=None):
    """
    Feature extraction function for the final feature set.

    Computes min, max, mean, and median of a target column (typically "ElImp"),
    together with max/mean, max/median, and mean/median ratios. The hour of day
    when the maximum value occurs is also stored.

    If `filtered` and `filter_val` are provided, the features are computed only
    for rows where `df[filtered] < filter_val`. For example, if `filtered` is
    "Tout" and `filter_val` is 10, the features are extracted only for time steps
    where the outdoor temperature is below 10 °C.

    Parameters
    ----------
    d : dict
        Dictionary to update with the computed features.
    df : pandas.DataFrame
        DataFrame containing the target column and optionally a filtering column.
    target : str
        Column name to extract features from.
    filtered : str, optional
        Column name used for filtering (default None).
    filter_val : any, optional
        Threshold value used for filtering.

    Returns
    -------
    dict
        Updated dictionary containing extracted features named:
        - {target}_max{filter_info}
        - {target}_mean{filter_info}
        - {target}_median{filter_info}
        - {target}_max_vs_mean{filter_info}
        - {target}_max_vs_median{filter_info}
        - {target}_mean_vs_median{filter_info}
        - {target}_max_hour{filter_info}
    """
    if filtered is not None and filter_val is not None:
        filtered_df = df[df[filtered] < filter_val]
    else:
        filtered_df = df
    max_value = filtered_df[target].max()
    mean_value = filtered_df[target].mean()
    median_value = filtered_df[target].median()
    max_vs_mean_value   = (max_value / mean_value)  if pd.notna(mean_value)  and mean_value  != 0 else np.nan
    max_vs_median_value = (max_value / median_value) if pd.notna(median_value) and median_value != 0 else np.nan
    mean_vs_median_value= (mean_value / median_value) if pd.notna(median_value) and median_value != 0 else np.nan
    
    
    if filtered is not None and filter_val is not None:
        filter_info = f"_{filtered}_below_{str(filter_val)}"
    else:
        filter_info = ""

    d[f"{target}_max{filter_info}"] = max_value
    d[f"{target}_mean{filter_info}"] = mean_value
    d[f"{target}_median{filter_info}"] = median_value
    d[f"{target}_max_vs_mean{filter_info}"] = max_vs_mean_value
    d[f"{target}_max_vs_median{filter_info}"] = max_vs_median_value
    d[f"{target}_mean_vs_median{filter_info}"] = mean_vs_median_value

    max_hour = filtered_df.loc[filtered_df[target] == max_value, "hour"].values[0]
    d[f"{target}_max_hour{filter_info}"] = int(max_hour) if pd.notna(max_hour) else np.nan

    return d


def add_daily_average_profile_features(d: dict, df: pd.DataFrame, filtered: str = None, filter_val=None) -> dict:
    """
    Feature extraction function for the final feature set.

    Computes features from the average daily load profile of a target column
    (typically "ElImp"). The average profile is obtained by averaging hourly
    values over all days. From this profile, the function extracts extrema,
    variability measures, characteristic hours, hour-to-hour changes, and
    the full 24-hour average profile. In addition, average daily values per
    weekday are computed.

    If `filtered` and `filter_val` are provided, the features are computed only
    for rows where `df[filtered] == filter_val`. For example, if `filtered` is
    "season" and `filter_val` is 1, the features are extracted only for winter
    months.

    Parameters
    ----------
    d : dict
        Dictionary to update with the computed features.
    df : pandas.DataFrame
        DataFrame containing at least the columns "ElImp", "hour", and "dayofweek".
    filtered : str, optional
        Column name used for filtering (default None).
    filter_val : any, optional
        Value used for filtering (default None).

    Returns
    -------
    dict
        Updated dictionary containing extracted features named:
        - daily_average_profile_max{filter_info} and daily_average_profile_max_hour{filter_info}
        - daily_average_profile_min{filter_info} and daily_average_profile_min_hour{filter_info}
        - daily_average_profile_second_max{filter_info} and hour
        - daily_average_median{filter_info}, daily_average_std{filter_info},
          daily_average_var{filter_info}, daily_average_max_over_min{filter_info}
        - daily_average_profile_max_change{filter_info} and hour
        - daily_average_profile_min_change{filter_info} and hour
        - daily_average_profile_hour_0–23{filter_info}
        - daily_average_elimp_day_0–6{filter_info}
    """
    if filtered is not None and filter_val is not None:
        df = df[df[filtered] == filter_val]
        filter_info = f"_{filtered}_{filter_val}"
    else:
        filter_info = ""

    hourly_avg = df.groupby("hour")["ElImp"].mean().reindex(range(24))
    average_daily_profile = pd.DataFrame({"hour": range(24), "ElImp": hourly_avg.values})
    average_daily_profile["ElImp_Derivative"] = average_daily_profile["ElImp"].diff()
    max_hour = int(average_daily_profile["ElImp"].idxmax())
    max_value = float(average_daily_profile.loc[max_hour, "ElImp"])
    min_hour = int(average_daily_profile["ElImp"].idxmin())
    min_value = float(average_daily_profile.loc[min_hour, "ElImp"])
    median_value = float(average_daily_profile["ElImp"].median())
    sorted_vals = average_daily_profile["ElImp"].sort_values(ascending=False)
    second_max_hour = int(sorted_vals.index[1]) if len(sorted_vals) > 1 else np.nan
    second_max_value = float(average_daily_profile.loc[second_max_hour, "ElImp"]) if len(sorted_vals) > 1 else np.nan
    max_change_hour = int(average_daily_profile["ElImp_Derivative"].idxmax())
    max_change = float(average_daily_profile.loc[max_change_hour, "ElImp_Derivative"])
    min_change_hour = int(average_daily_profile["ElImp_Derivative"].idxmin())
    min_change = float(average_daily_profile.loc[min_change_hour, "ElImp_Derivative"])

    before_max_hour = (max_hour - 1) % 24
    after_max_hour = (max_hour + 1) % 24

    d[f"daily_average_profile_second_max{filter_info}"] = second_max_value
    d[f"daily_average_profile_second_max_hour{filter_info}"] = second_max_hour
    d[f"daily_average_median{filter_info}"] = median_value
    d[f"daily_average_std{filter_info}"] = float(average_daily_profile["ElImp"].std())
    d[f"daily_average_var{filter_info}"] = float(average_daily_profile["ElImp"].var())
    d[f"daily_average_max_over_min{filter_info}"] = (max_value / min_value) if min_value != 0 else np.nan
    d[f"daily_average_profile_max{filter_info}"] = max_value
    d[f"daily_average_profile_max_hour{filter_info}"] = max_hour
    d[f"daily_average_profile_max_before{filter_info}"] = float(average_daily_profile.loc[before_max_hour, "ElImp"])
    d[f"daily_average_profile_max_after{filter_info}"] = float(average_daily_profile.loc[after_max_hour, "ElImp"])
    d[f"daily_average_profile_min{filter_info}"] = min_value
    d[f"daily_average_profile_min_hour{filter_info}"] = min_hour
    d[f"daily_average_profile_max_change_hour{filter_info}"] = max_change_hour
    d[f"daily_average_profile_min_change_hour{filter_info}"] = min_change_hour
    d[f"daily_average_profile_max_change{filter_info}"] = max_change
    d[f"daily_average_profile_min_change{filter_info}"] = min_change

    for i in range(24):
        d[f"daily_average_profile_hour_{i}{filter_info}"] = float(average_daily_profile.loc[i, "ElImp"])

    daily_avg = df.groupby("dayofweek")["ElImp"].mean().reindex(range(7))
    for i in range(7):
        d[f"daily_average_elimp_day_{i}{filter_info}"] = float(daily_avg.iloc[i]) if not np.isnan(daily_avg.iloc[i]) else np.nan

    return d


def add_daily_standard_load_profiles_correlation_features(slp_dict, d, df):
    """
    Feature extraction function for the final feature set.

    Computes the Pearson correlation between the building’s average daily load
    profile (typically based on "ElImp") and a set of standard load profiles
    (SLPs). The average daily profile is obtained by averaging hourly values
    over all days.

    Parameters
    ----------
    slp_dict : dict
        Dictionary of standard load profiles. Each entry must contain an
        "ElImp" array with 24 hourly values.
    d : dict
        Dictionary to update with the computed features.
    df : pandas.DataFrame
        DataFrame containing the columns "ElImp" and "hour" (0–23).

    Returns
    -------
    dict
        Updated dictionary containing extracted features named:
        - correlation_daily_SLP_<profile_name>
    """
    hourly_avg = df.groupby("hour")["ElImp"].mean()
    average_daily_profile = pd.DataFrame({"hour": range(24), "ElImp": hourly_avg.values})

    for key in slp_dict.keys():
        correlation = average_daily_profile["ElImp"].corr(slp_dict[key]["ElImp"])
        d[f"correlation_daily_SLP_{key}"] = correlation

    return d


def add_EV_specific_features(d: dict, df: pd.DataFrame) -> dict:
    """
    Feature extraction function for the final feature set.

    Computes features designed to capture abrupt changes in electricity
    consumption that are characteristic of household EV charging. The features
    are based on absolute hour-to-hour changes in the target column
    (typically "ElImp") and aim to detect large charging-related load increases.

    The function stores the largest individual changes as well as the average
    of the 50 largest changes.

    Parameters
    ----------
    d : dict
        Dictionary to update with the computed features.
    df : pandas.DataFrame
        DataFrame containing the column "ElImp".

    Returns
    -------
    dict
        Updated dictionary containing extracted features named:
        - ElImp_0_max_change ... ElImp_9_max_change
        - average_first_50
    """
    if "ElImp" not in df or df["ElImp"].dropna().empty:
        for i in range(10):
            d[f"ElImp_{i}_max_change"] = np.nan
        d["average_first_50"] = np.nan
        return d

    ElImp_diff = df["ElImp"].diff().abs().dropna().sort_values(ascending=False)
    for i in range(min(10, len(ElImp_diff))):
        d[f"ElImp_{i}_max_change"] = float(ElImp_diff.iloc[i])
    d["average_first_50"] = float(ElImp_diff.head(50).mean()) if len(ElImp_diff) > 0 else np.nan
    return d


def add_ETCurve_features(d: dict, df: pd.DataFrame, res: str = "h") -> dict:
    """
    Feature extraction function for the final feature set.

    Computes ET-curve features from the relationship between outdoor temperature
    ("Tout") and electricity import (typically "ElImp"). A 2-segment piecewise
    linear model with one change point is fitted to ElImp as a function of Tout.
    The change point temperature (CPT) and the slopes on each side are stored,
    together with residual statistics below and above the CPT.

    If `res` is "d", the data are resampled to daily values (mean Tout and sum
    ElImp) before fitting. If there are too few valid points, NaNs are stored. 
    If 'res' is "h", the ET-curve is extracted from the hourly time series (default).

    Parameters
    ----------
    d : dict
        Dictionary to update with the computed features.
    df : pandas.DataFrame
        DataFrame containing the columns "Tout", "ElImp", and "TimeStamp".
    res : str, optional
        Time resolution used for fitting: "h" for hourly (default) or "d" for daily.

    Returns
    -------
    dict
        Updated dictionary containing extracted features named:
        - EH_features_{res}_CPT
        - EH_features_{res}_B0
        - EH_features_{res}_B1
        - EH_features_{res}_B2
        - EH_features_{res}_std_below, EH_features_{res}_std_above
        - EH_features_{res}_mean_abs_below, EH_features_{res}_mean_abs_above
        - EH_features_{res}_min_below, EH_features_{res}_max_below
        - EH_features_{res}_min_above, EH_features_{res}_max_above
    """
    df = df[["Tout", "ElImp", "TimeStamp"]].dropna()
    if df.empty:
        for k in ["CPT","B0","B1","B2","std_below","std_above","mean_abs_below","mean_abs_above","min_below","max_below","min_above","max_above"]:
            d[f"EH_features_{res}_{k}"] = np.nan
        return d

    if res == "d":
        df = (df.set_index("TimeStamp")
                .resample("D")
                .agg({"Tout": "mean", "ElImp": "sum"})
                .dropna())
        if df.empty:
            for k in ["CPT","B0","B1","B2","std_below","std_above","mean_abs_below","mean_abs_above","min_below","max_below","min_above","max_above"]:
                d[f"EH_features_{res}_{k}"] = np.nan
            return d

    T = df["Tout"].values
    E = df["ElImp"].values
    if len(E) < 3:
        for k in ["CPT","B0","B1","B2","std_below","std_above","mean_abs_below","mean_abs_above","min_below","max_below","min_above","max_above"]:
            d[f"EH_features_{res}_{k}"] = np.nan
        return d

    model = pwlf.PiecewiseLinFit(T, E)
    breaks = model.fit(2)
    CPT = float(breaks[1])
    below_mask = T <= CPT
    above_mask = T > CPT
    T_below, E_below = T[below_mask], E[below_mask]
    T_above, E_above = T[above_mask], E[above_mask]

    # Guard against degenerate split
    if len(T_below) < 2 or len(T_above) < 2:
        for k in ["CPT","B0","B1","B2","std_below","std_above","mean_abs_below","mean_abs_above","min_below","max_below","min_above","max_above"]:
            d[f"EH_features_{res}_{k}"] = np.nan
        d[f"EH_features_{res}_CPT"] = CPT
        return d

    reg_below = LinearRegression().fit(T_below.reshape(-1, 1), E_below)
    B1 = float(reg_below.coef_[0])
    B0 = float(reg_below.intercept_ + B1 * CPT)
    reg_above = LinearRegression().fit(T_above.reshape(-1, 1), E_above)
    B2 = float(reg_above.coef_[0])
    eps_below = E_below - reg_below.predict(T_below.reshape(-1, 1))
    eps_above = E_above - reg_above.predict(T_above.reshape(-1, 1))

    d[f"EH_features_{res}_CPT"] = CPT
    d[f"EH_features_{res}_B0"] = B0
    d[f"EH_features_{res}_B1"] = B1
    d[f"EH_features_{res}_B2"] = B2
    d[f"EH_features_{res}_std_below"] = float(np.std(eps_below))
    d[f"EH_features_{res}_std_above"] = float(np.std(eps_above))
    d[f"EH_features_{res}_mean_abs_below"] = float(np.mean(np.abs(eps_below)))
    d[f"EH_features_{res}_mean_abs_above"] = float(np.mean(np.abs(eps_above)))
    d[f"EH_features_{res}_min_below"] = float(np.min(eps_below))
    d[f"EH_features_{res}_max_below"] = float(np.max(eps_below))
    d[f"EH_features_{res}_min_above"] = float(np.min(eps_above))
    d[f"EH_features_{res}_max_above"] = float(np.max(eps_above))
    return d


def add_AutoCorrelation_features(d: dict, df: pd.DataFrame) -> dict:
    """
    Feature extraction function for the final feature set.

    Computes autocorrelation features of the energy signal (typically "ElImp")
    for time lags 1 to 10. Autocorrelation coefficients are computed on the
    forward-filled signal to handle missing values. If there are too few valid
    data points, NaNs are stored.

    Parameters
    ----------
    d : dict
        Dictionary to update with the computed features.
    df : pandas.DataFrame
        DataFrame containing the column "ElImp".

    Returns
    -------
    dict
        Updated dictionary containing extracted features named:
        - AC_feature_1 ... AC_feature_10
    """
    if "ElImp" not in df or df["ElImp"].dropna().empty:
        for lag in range(1, 11):
            d[f"AC_feature_{lag}"] = np.nan
        return d

    series = df["ElImp"].ffill()
    for lag in range(1, 11):
        d[f"AC_feature_{lag}"] = float(series.autocorr(lag))
    return d


def add_Spikes_features(d: dict, df: pd.DataFrame) -> dict:
    """
    Feature extraction function for the final feature set.

    Computes spike-related features from absolute hour-to-hour changes in the
    electricity signal (typically "ElImp"). Here, a spike is defined as the
    absolute difference between consecutive ElImp values. The spike series is
    sorted from largest to smallest to separate rare, large changes from more
    common small changes.

    A 2-segment piecewise linear model with one change point (CPT) is fitted to
    the sorted spike magnitudes. The CPT separates the few largest spikes from
    the remaining smaller spikes. Slopes below and above the CPT (B1 and B2)
    describe how spike magnitudes decrease in each region.

    Additional features describe how large the spikes are, using
    ratios relative to the mean and median spike size and the share of spikes
    above and below the CPT.

    Parameters
    ----------
    d : dict
        Dictionary to update with the computed features.
    df : pandas.DataFrame
        DataFrame containing the column "ElImp".

    Returns
    -------
    dict
        Updated dictionary containing extracted features named:
        - Spikes_B0
        - Spikes_B1
        - Spikes_B2
        - Spikes_B0_vs_mean
        - Spikes_B0_vs_median
        - Spikes_share_above_CPT
        - Spikes_share_below_CPT
    """
    spikes = df["ElImp"].diff().abs().dropna().sort_values(ascending=False).reset_index(drop=True)
    if spikes.empty:
        d.update({
            "Spikes_B0": np.nan, "Spikes_B1": np.nan, "Spikes_B2": np.nan,
            "Spikes_B0_vs_mean": np.nan, "Spikes_B0_vs_median": np.nan,
            "Spikes_share_above_CPT": np.nan, "Spikes_share_below_CPT": np.nan
        })
        return d

    E = spikes.values
    T = spikes.index
    model = pwlf.PiecewiseLinFit(T, E)
    breaks = model.fit(2)
    CPT = breaks[1]

    T_below_CPT = T[T <= CPT]
    E_below_CPT = E[T <= CPT]
    T_above_CPT = T[T > CPT]
    E_above_CPT = E[T > CPT]

    reg_below = LinearRegression().fit(T_below_CPT.values.reshape(-1, 1), E_below_CPT)
    B1 = float(reg_below.coef_[0])
    B0 = float(reg_below.intercept_ + B1 * CPT)

    reg_above = LinearRegression().fit(T_above_CPT.values.reshape(-1, 1), E_above_CPT)
    B2 = float(reg_above.coef_[0])

    d["Spikes_B0"] = B0
    d["Spikes_B1"] = B1
    d["Spikes_B2"] = B2
    d["Spikes_B0_vs_mean"] = float(B0 / E.mean()) if len(E) else np.nan
    d["Spikes_B0_vs_median"] = float(B0 / np.median(E)) if len(E) else np.nan
    d["Spikes_share_above_CPT"] = float(len(E_above_CPT) / len(E)) if len(E) else np.nan
    d["Spikes_share_below_CPT"] = float(len(E_below_CPT) / len(E)) if len(E) else np.nan
    return d


def add_temperature_correlation_features(d,df):
    """
    Feature extraction function for the final feature set.

    Computes correlation features between electricity import ("ElImp")
    and outdoor temperature ("Tout") columns. Both Spearman (rank-based) and Pearson
    (linear) correlation coefficients are calculated.

    Parameters
    ----------
    d : dict
        Dictionary to update with the computed features.
    df : pandas.DataFrame
        DataFrame containing the columns "ElImp" and "Tout".

    Returns
    -------
    dict
        Updated dictionary containing extracted features named:
        - Corr_Tout_Spearman
        - Corr_Tout_Pearson
    """
    corr_spearman = df["ElImp"].corr(df["Tout"], method="spearman")
    corr_pearson = df["ElImp"].corr(df["Tout"], method="pearson")
    d["Corr_Tout_Spearman"] = corr_spearman
    d["Corr_Tout_Pearson"] = corr_pearson
    return d


def get_extracted_features_from_building_ts(ts, building_id="building", savedir=None, filename="classification_features", normalize = True):
    """
    Main feature extraction function for a single building.

    Extracts a fixed set of features used for classification of building type,
    including building category and heating type, from a single building time
    series in TREASURE format. The function produces one feature vector per
    building, independent of the time series duration.

    The input data are prepared, calendar features are derived, electricity
    import ("ElImp") is optionally normalized, and a sequence of feature
    extraction functions is applied.

    The extracted features form a single feature vector of fixed size
    (1 × ~180 features) for each building.

    Parameters
    ----------
    ts : pandas.DataFrame
        Time series data for a single building in TREASURE format.
    building_id : str, optional
        Identifier (building name/ID) for the building (default "building").
    savedir : str, optional
        Directory where the extracted features are saved. If None, the results
        are not written to disk.
    filename : str, optional
        Base name of the output file (default "classification_features").
        The file is saved as a .txt file using ';' as separator.
    normalize : bool, optional
        Whether to normalize the "ElImp" column before feature extraction
        (default True).

    Returns
    -------
    pandas.DataFrame
        One-row DataFrame containing the extracted features for the given
        building, indexed by building ID.
    """

    ts = ts.reset_index(drop=False)

    if "ElImp" not in ts.columns and "ElTot" in ts.columns:
        ts["ElImp"] = ts["ElTot"]
    
    elif "ElImp" in ts.columns and "ElTot" in ts.columns:
        if ts["ElTot"].sum() > ts["ElImp"].sum():
            ts["ElImp"] = ts["ElTot"]
    elif "ElImp" not in ts.columns and "ElTot" not in ts.columns and "X" in ts.columns:
        ts["ElImp"] = ts["X"]
    
    ts = add_date_features(ts)
    ts = ts[ts["ElImp"].notna() & (ts["ElImp"] != 0)]

    building_features = {}
    building_features["ElImp_actual_max"] = ts['ElImp'].max()
    building_features["ElImp_actual_mean"] = ts['ElImp'].mean() 
    try:
        if normalize:         
            ts_normalized = df_normalizer(ts, ["ElImp"])
        else: 
            ts_normalized = ts

        print(f"Extracting classification features from {building_id}")
        building_features = add_min_max_mean_features(building_features, ts_normalized, "ElImp")
        for season in [1, 2, 3, 4]:
            building_features = add_min_max_mean_features(building_features, ts_normalized, "ElImp", "season", season)
        for weekend in [1, 0]:
            building_features = add_min_max_mean_features(building_features, ts_normalized, "ElImp", "weekend", weekend)
        building_features = add_daytime_features(building_features, ts_normalized)
        building_features = add_min_max_mean_below_feature(building_features, ts_normalized, "ElImp", "Tout", 10)
        building_features = add_daily_average_profile_features(building_features, ts_normalized)
        building_features = add_daily_standard_load_profiles_correlation_features(daily_slp, building_features, ts_normalized)
        building_features['Tout_at_ElImp_max'] = ts_normalized.at[ts_normalized['ElImp'].idxmax(), 'Tout']
        building_features['ElImp_at_Tout_min'] = ts_normalized.at[ts_normalized['Tout'].idxmin(), 'ElImp']
        building_features['ElImp_mean_vs_winter_mean'] = building_features["ElImp_mean"] / building_features["ElImp_mean_season_is_1"]
        building_features['ElImp_mean_vs_summer_mean'] = building_features["ElImp_mean"] / building_features["ElImp_mean_season_is_3"]
        building_features['ElImp_winter_mean_vs_summer_mean'] = building_features["ElImp_mean_season_is_1"] / building_features["ElImp_mean_season_is_3"]
        building_features['ElImp_max_vs_summer_max'] = building_features["ElImp_max"] / building_features["ElImp_max_season_is_3"]
        building_features = add_EV_specific_features(building_features, ts)
        building_features = add_ETCurve_features(building_features, ts, "h")
        building_features = add_ETCurve_features(building_features, ts, "d")
        building_features = add_AutoCorrelation_features(building_features, ts)
        building_features = add_Spikes_features(building_features, ts)
        building_features = add_temperature_correlation_features(building_features, ts)

    except Exception:
        print(f"Error while processing {building_id}")

    building_features_df = pd.DataFrame.from_dict(
        building_features, orient='index', columns=[building_id]
    ).T
    building_features_df.index.name = 'building_id'
    
    building_features_df = building_features_df.replace([np.inf, -np.inf], np.nan).fillna(0)
    
    if savedir:
        import os
        os.makedirs(savedir, exist_ok=True)
        filename = filename.strip()
        if filename.lower().endswith(".csv"):
            filename = filename[:-4] + ".txt"
        elif not filename.lower().endswith(".txt"):
            filename += ".txt"
        building_features_df.to_csv(os.path.join(savedir, "features_" + filename), sep=";")
    
    return building_features_df



def get_extracted_features_from_building_dict(ts_dict, savedir=None, filename="classification_features", normalize = True):
    """
    Main feature extraction function for multiple buildings.
    
    Iterates over a dictionary of building time series (one DataFrame per building)
    and calls `get_extracted_features_from_building_ts` for each building. The
    resulting fixed-size feature vectors are combined into a single DataFrame and
    are used for classification of building category and heating type, independent
    of time series duration.
    
    Each building contributes one row in the output DataFrame, corresponding to
    one feature vector (~180 features).
    
    Parameters
    ----------
    ts_dict : dict
        Dictionary mapping building IDs to TREASURE-format DataFrames.
    savedir : str, optional
        Directory where the extracted features are saved.
    filename : str, optional
        Base name of the output file (default "classification_features").
    normalize : bool, optional
        Whether to normalize the "ElImp" column before feature extraction.
    
    Returns
    -------
    pandas.DataFrame
        DataFrame containing one row of features per building.
    """
    rows = []
    for building, df in ts_dict.items():
        try:
            feats = get_extracted_features_from_building_ts(df, building,normalize = normalize)
            feats["building_id"] = building
            rows.append(feats)
        except Exception:
            # Skip buildings that cannot be processed
            continue

    if not rows:
        return pd.DataFrame()  # handle case with no valid rows

    building_features_df = pd.concat(rows, ignore_index=True).set_index("building_id")

    if savedir:
        import os
        os.makedirs(savedir, exist_ok=True)

        filename = filename.strip()
        if filename.lower().endswith(".csv"):
            filename = filename[:-4] + ".txt"
        elif not filename.lower().endswith(".txt"):
            filename += ".txt"

        building_features_df.to_csv(os.path.join(savedir, "features_" + filename), sep=";")

    return building_features_df


def get_extracted_features_from_file(path, savedir=None, filename="classification_features", normalize = True):
    """
    Main feature extraction function for a single building file.

    Loads a TREASURE-format building file and calls
    `get_extracted_features_from_building_ts` to extract a fixed-size feature
    vector used for classification of building category and heating type. The
    result is returned as a one-row DataFrame.

    Parameters
    ----------
    path : str
        Path to a TREASURE-format building file.
    savedir : str, optional
        Directory where the extracted features are saved.
    filename : str, optional
        Base name of the output file (default "classification_features").
    normalize : bool, optional
        Whether to normalize the "ElImp" column before feature extraction.

    Returns
    -------
    pandas.DataFrame
        One-row DataFrame containing extracted features for the building.
    """
    building_name = os.path.basename(path)
    building_name = building_name.replace('.txt', '').replace('.csv', '')
    meta, ts = load_ts_data(path)
    building_features = get_extracted_features_from_building_ts(ts,building_name,savedir,filename,normalize = normalize)

    return building_features

def get_extracted_features_from_folder(path, savedir=None, filename="classification_features", normalize = True):
    """
    Main feature extraction function for a folder of building files.
    
    Loads TREASURE-format building files from a folder and calls
    `get_extracted_features_from_building_dict` to extract fixed-size feature
    vectors for all buildings. The features are used for classification of
    building category and heating type and are independent of time series
    duration.
    
    Parameters
    ----------
    path : str
        Path to a folder containing TREASURE-format building files.
    savedir : str, optional
        Directory where the extracted features (and metadata) are saved.
    filename : str, optional
        Base name of the output file (default "classification_features").
    normalize : bool, optional
        Whether to normalize the "ElImp" column before feature extraction.
    
    Returns
    -------
    pandas.DataFrame
        DataFrame containing one row of extracted features per building.
    """
    meta_dict, ts_dict = load_ts_data(path)
    meta_df = pd.DataFrame.from_dict(meta_dict, orient='index')
    building_features_df = get_extracted_features_from_building_dict(ts_dict,normalize = normalize)

    if savedir:
        import os
        os.makedirs(savedir, exist_ok=True)
        filename = filename.strip()
        if filename.lower().endswith(".csv"):
            filename = filename[:-4] + ".txt"
        elif not filename.lower().endswith(".txt"):
            filename += ".txt"

        building_features_df.to_csv(os.path.join(savedir, "features_" + filename), sep=";")
        meta_df.to_csv(os.path.join(savedir, "meta_" + filename), sep=";")

    return building_features_df





def get_true_electric_SH_label(heating_list, heating_categories, building_category):
    """
    Extract ground-truth labels for building classification.

    Determines whether a building uses electricity for space heating based on
    heating source information from metadata stored in a TREASURE file. This
    function is used by `get_true_building_label` to assign the heating type
    component of the building label. The result is used during training and
    validation.

    Parameters
    ----------
    heating_list : list
        List of space-heating sources from building metadata.
    heating_categories : pandas.DataFrame
        Lookup table defining which heating sources are electric.
    building_category : str
        Building category (e.g. "Apt", "Apb", "SFH").

    Returns
    -------
    bool
        True if the building uses electric space heating, otherwise False.
    """
    if building_category == "Apt":
        is_electric = heating_categories.loc[heating_list, 'section_electric'].any()
    else:
        is_electric = heating_categories.loc[heating_list, 'building_electric'].any()

    return bool(is_electric)

def get_true_building_label(meta, ts):
    """
    Extract ground-truth labels for building classification.

    Combines building category and space-heating type into a single true label
    using metadata from a TREASURE file. The heating type is determined using
    `get_true_electric_SH_label`. The resulting label is used during training
    and validation of classification models.

    Examples
    --------
    Apartment with electric space heating:
        meta["building_category"] = "Apt" (apartment)
        meta["sh_heat_source"] = "ASHP" (air-source heat pump)
        -> "Apt_EH"

    Single-family house with non-electric heating:
        meta["building_category"] = "Off" (office)
        meta["sh_heat_source"] = "DH" (district heating)
        -> "Off_NEH"

    Unknown or missing metadata:
        -> "ukn_ukn"

    Parameters
    ----------
    meta : dict
        Metadata dictionary for a single building.
    ts : pandas.DataFrame
        Building time series data, used to resolve ambiguous building categories.

    Returns
    -------
    str
        True building label in the format "<building_category>_<heating_type>".
    """
    # Identify the building category from the metadata
    try:
        building_category = meta.get('building_category', 'ukn')
        if building_category == "Apt" and ts["ElImp"].max() > 25000:
            building_category = "Apb"
        if building_category in ["Non", "Ukn", np.nan, ""]:
            building_category = "ukn"
    except Exception:
        building_category = "ukn"

    # Determine heating category (EH or NEH)
    try:
        heating_sources = [s.strip() for s in meta["sh_heat_source"].split(',')]
        if get_true_electric_SH_label(heating_sources, heating_categories, building_category):
            heating_category = "EH"
        else:
            heating_category = "NEH"
    except Exception:
        heating_category = "ukn"

    return f"{building_category}_{heating_category}"


def get_true_building_label_dict(meta_dict, ts_dict):
    """
    Extract ground-truth labels for building classification.

    Applies `get_true_building_label` to multiple buildings using metadata from
    TREASURE files and returns the corresponding ground-truth labels. These
    labels are used during training and validation of classification models.

    Parameters
    ----------
    meta_dict : dict
        Dictionary mapping building IDs to metadata dictionaries.
    ts_dict : dict
        Dictionary mapping building IDs to building time series DataFrames.

    Returns
    -------
    pandas.DataFrame
        DataFrame indexed by building ID with one column:
        - 'true_building_label'
    """
    labels = {}

    for building_id in meta_dict.keys():
        label = get_true_building_label(meta_dict[building_id], ts_dict[building_id])
        labels[building_id] = label

    return pd.DataFrame.from_dict(labels, orient="index", columns=["true_building_label"])

