# -----------------------------------------------------------------------------
# Author: Synne Krekling Lien
# Contact: synne.lien@sintef.no
# Date: 23.02.2026
# Repository: https://github.com/synnekreklinglien/COF-tool
#
# Example workflow for training and testing the CofClassification module for
# supervised classification of building category and heating type using
# electricity smart meter data.
#
# The CofClassification module performs building-level classification based on
# hourly smart meter time series. Each building is assigned a class representing:
# 1) the building category, and
# 2) the dominant heating type (electric or non-electric).
#
# The classification output is a short code encoding both properties.
#
# Building category codes:
#   Apb = apartment block, Apt = apartment, Cab = cabin, Hou = house,
#   Hsp = hospital, Htl = hotel, Kdg = kindergarten, Nsh = nursing home,
#   Off = office, Sch = school, Shp = shop, Uni = university
#
# Heating type codes:
#   NEH = non-electric heating, EH = electric heating
#
# This script demonstrates a complete end-to-end classification workflow:
# 1) Load building time series data and extract classification features
# 2) Split buildings into training and test sets at building level
# 3) Train a classification model and evaluate its performance
# 4) Apply the trained model to a new building time series using:
#    - the in-memory model
#    - a newly trained model loaded explicitly from file
#    - the trained default model stored in the repository. 
#
# Example data source:
# COFACTOR-Drammen dataset
# https://doi.org/10.1038/s41597-025-04708-3
#
# Note:
# Due to the small number of buildings in the example dataset, the resulting
# performance metrics are not representative of real-world classification accuracy.
# -----------------------------------------------------------------------------

# -----------------------------------------------------------------------------
# Imports
# -----------------------------------------------------------------------------
import sys
from pathlib import Path
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]  # .../COF-tool
sys.path.insert(0, str(PROJECT_ROOT))

import modules.CofClassification.lib.ReadtreaData as treaData
import modules.CofClassification.lib.CofClassifyClassify as CofClassifyClassify
import modules.CofClassification.lib.CofClassifyFeatureExtraction as CofClassifyFeatureExtraction


# -----------------------------------------------------------------------------
# Load example classification input data and extract features and labels from the files.
# -----------------------------------------------------------------------------
file_path = PROJECT_ROOT / "examples" / "example_files" / "input_files"

# Read building metadata and time series from disk
buildings_meta, buildings_ts = treaData.readAllcsvsFromFolder(
    f"{file_path}/",
    ext="txt",
)

# Extract classification features for each building time series.
buildings_FE = CofClassifyFeatureExtraction.get_extracted_features_from_building_dict(
    buildings_ts
)

# Retrieve the true building class labels from each building file.
buildings_labels = CofClassifyFeatureExtraction.get_true_building_label_dict(
    buildings_meta,
    buildings_ts,
)

# -----------------------------------------------------------------------------
# Shuffle and split rows into train and test. X = feature extracted dataset, y = true labels (building category and heating type)
# -----------------------------------------------------------------------------

common_idx = buildings_FE.index.intersection(buildings_labels.index)
X = buildings_FE.loc[common_idx]
y = buildings_labels.loc[common_idx]
rng = np.random.default_rng(42)  
idx = X.index.to_numpy()
rng.shuffle(idx)

train_share = 0.5
split_idx = int(train_share * len(idx))

train_idx = idx[:split_idx]
test_idx = idx[split_idx:]

X_train = X.loc[train_idx]
X_test = X.loc[test_idx]
y_train = y.loc[train_idx]
y_test = y.loc[test_idx]


# In the case with little example data: remove unseen labels from the test set
# Classification models cannot predict labels that were never observed
# during training. Any such samples are removed from the test set.

label_col = y.columns[0]
seen_labels = set(y_train[label_col].unique())
mask_seen = y_test[label_col].isin(seen_labels)
X_test = X_test.loc[mask_seen]
y_test = y_test.loc[mask_seen]


# -----------------------------------------------------------------------------
# Train the classification model and save it to disk
# -----------------------------------------------------------------------------
model, label_encoder = CofClassifyClassify.train_classification_model(
    X_train=X_train,
    y_train=y_train,
    filename="example_classifier.joblib",
)


# -----------------------------------------------------------------------------
# Evaluate model performance on the test set by making a confusion matrix and classification report are intended for
# beware that the poor results in this example is due to the low number of buildings in the training and test set. 
# -----------------------------------------------------------------------------
classification_report = CofClassifyClassify.evaluate_classification_model(
    model,
    label_encoder,
    X_test,
    y_test,
    plot=True,
)


# -----------------------------------------------------------------------------
# Apply the trained model to a new building time series
# -----------------------------------------------------------------------------
file_path = PROJECT_ROOT / "examples" / "example_files" / "Cof_example_file.txt"
building_meta, building_df = treaData.getTreaCsv(str(file_path))


# Apply the trained model (from memory) to the new example building
probs_in_memory = CofClassifyClassify.predict_class_probabilities_ts(
    building_df,
    building_id="building - in memory",
    model=model,
    label_encoder=label_encoder,
)

fig_in_memory = CofClassifyClassify.plot_building_class_probabilities(
    probs_in_memory, title = "Class probabilities, new model (in memory)"
)


# Load model from disk and aply to the example building. 
example_model_path = PROJECT_ROOT / "examples" / "example_classifier.joblib"

probs_in_file = CofClassifyClassify.predict_class_probabilities_ts(
    building_df,
    building_id="building - model from file",
    model_path=example_model_path,
)

fig_in_file = CofClassifyClassify.plot_building_class_probabilities(
    probs_in_file, title = "Class probabilities, new saved model"
)


# For comparison, apply the trained default model to the example building. 
probs_default = CofClassifyClassify.predict_class_probabilities_ts(
    building_df,
    building_id="building - default model",
)

fig_default = CofClassifyClassify.plot_building_class_probabilities(
    probs_default, title = "Class probabilities, default model"
)