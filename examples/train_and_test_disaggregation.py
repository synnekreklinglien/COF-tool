# -----------------------------------------------------------------------------
# Author: Synne Krekling Lien
# Contact: synne.lien@sintef.no
# Date: 23.02.2026
# Repository: https://github.com/synnekreklinglien/COF-tool
#
# Example workflow for training and testing the CofDisaggregation module
# for supervised estimation of electricity used for heating from aggregated
# electric smart meter time series data.
#
# The script shows how to:
# 1) Load example building time series data (Trea format) and clean inputs
# 2) Extract disaggregation features and prepare the target variable (ElBoil)
# 3) Split buildings into training and test sets at building level
# 4) Train a disaggregation model and evaluate predictions on the test set
#
# Example data sources:
# - COFACTOR-Drammen dataset (buildings: 6404, 6414, 6418)
#   https://doi.org/10.1038/s41597-025-04708-3
# - COFACTOR-SBHUB-Oslo dataset (buildings: 8112, 8138)
#   https://doi.org/10.1016/j.dib.2025.112288
# -----------------------------------------------------------------------------

# -----------------------------------------------------------------------------
# Imports
# -----------------------------------------------------------------------------
import sys
import random
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]  # .../COF-tool
sys.path.insert(0, str(PROJECT_ROOT))

import modules.CofClassification.lib.ReadtreaData as treaData
import modules.CofDisaggregation.lib.CofDisaggregationDisaggregate as CofDisaggregationDisaggregate
import modules.CofDisaggregation.lib.CofDisaggregationFeatureExtraction as CofDisaggregationFeatureExtraction


# -----------------------------------------------------------------------------
# Load example data and clean inputs
#
# In this example, all buildings are schools with electric boilers.
# ElImp (imported electricity) is an input variable, while ElBoil
# (electricity use for the electric boiler) is the target variable.
# -----------------------------------------------------------------------------
file_path = PROJECT_ROOT / "examples" / "example_files" / "input_files"
buildings_meta, buildings_ts = treaData.readAllcsvsFromFolder(f"{file_path}/", ext="txt")

# Clean input data:
# - Replace NaN with 0
# - Clip negative values to 0 for ElImp and ElBoil
for building, df in buildings_ts.items():
    for col in ["ElImp", "ElBoil"]:
        if col in df.columns:
            df[col] = df[col].fillna(0).clip(lower=0)


# -----------------------------------------------------------------------------
# Feature extraction and target preparation
#
# - Extract features for all buildings (based on the original data, e.g., Tout, ElImp, timestamp)
# - Store ElBoil as the target by renaming it to "Y"
# -----------------------------------------------------------------------------
buildings_FE = CofDisaggregationFeatureExtraction.get_extracted_features_from_building_dict(buildings_ts)

buildings_Y = {
    building: df["ElBoil"].rename("Y")
    for building, df in buildings_ts.items()
    if "ElBoil" in df.columns
}


# -----------------------------------------------------------------------------
# Train/test split (by buildings)
#
# Shuffle buildings, then split into train and test sets for both features and targets.
# -----------------------------------------------------------------------------
buildings = list(set(buildings_FE) & set(buildings_Y))
random.shuffle(buildings)

train_share = 0.6
split_idx = int(train_share * len(buildings))

train_buildings = buildings[:split_idx]
test_buildings = buildings[split_idx:]

buildings_FE_train = {b: buildings_FE[b] for b in train_buildings}
buildings_FE_test = {b: buildings_FE[b] for b in test_buildings}

buildings_Y_train = {b: buildings_Y[b] for b in train_buildings}
buildings_Y_test = {b: buildings_Y[b] for b in test_buildings}


# -----------------------------------------------------------------------------
# Train the disaggregation model on the training set
#
# The model used in this example is CatBoost (see module implementation).
# -----------------------------------------------------------------------------
disaggregation_model = CofDisaggregationDisaggregate.train_disaggregation_model(
    buildings_FE_train,
    buildings_Y_train,
)


# -----------------------------------------------------------------------------
# Predict on the test set (one building at a time)
# -----------------------------------------------------------------------------
preds_test = {}
for building in buildings_Y_test:
    preds_test[building] = CofDisaggregationDisaggregate.disaggregate_feature_extracted_building(
        buildings_FE_test[building],
        disaggregation_model,
    )


# -----------------------------------------------------------------------------
# Evaluate performance and plot measured vs predicted heating electricity
# -----------------------------------------------------------------------------
performance_test = {}
for building in buildings_Y_test:
    performance_test[building] = CofDisaggregationDisaggregate.evaluate_disaggregation(
        buildings_Y_test[building],
        preds_test[building],
        True,       # make_plots
        building,   # building_id
    )