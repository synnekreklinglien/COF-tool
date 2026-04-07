# -----------------------------------------------------------------------------
# Author: Synne Krekling Lien
# Contact: synne.lien@sintef.no
# Date: 23.02.2026
# Repository: https://github.com/synnekreklinglien/COF-tool
#
# Example workflow demonstrating combined use of the CofClassification and
# CofDisaggregation modules for building-level analysis based on electric
# smart meter time series data.
#
# The script shows how to:
# 1) Load a building time series file in Trea format
# 2) Classify building category and heating type using a trained classification model
# 3) Disaggregate electricity use for heating using a trained disaggregation model
# -----------------------------------------------------------------------------

# ------------------------------------------------------------------
# Imports
# ------------------------------------------------------------------

import sys
from pathlib import Path
import matplotlib.pyplot as plt

PROJECT_ROOT = Path(__file__).resolve().parents[1]  # ...\COF-tool
sys.path.insert(0, str(PROJECT_ROOT))

import modules.CofClassification.lib.ReadtreaData as treaData
import modules.CofClassification.lib.CofClassifyClassify as CofClassifyClassify
import modules.CofDisaggregation.lib.CofDisaggregationDisaggregate as CofDisaggregationDisaggregate
import modules.CofDisaggregation.lib.CofDisaggregationFeatureExtraction as CofDisaggregationFeatureExtraction


# ------------------------------------------------------------------
# 1) Load building data and investigate
# ------------------------------------------------------------------
# This section loads the example file into:
# - building_meta: metadata about the building
# - building_df:   Hourly time series dataframe: Must include a timestamp and columns for imported electricity (ElImp) and outdoor temperature (Tout)
# ElImp (imported electricity) is plotted over time.
# ------------------------------------------------------------------

file_path = PROJECT_ROOT / "examples" / "example_files" / "Cof_example_file.txt"
if not file_path.exists():
    raise FileNotFoundError(f"Example file not found: {file_path}")

building_meta, building_df = treaData.getTreaCsv(str(file_path))

plt.figure(figsize=(6, 4))
plt.plot(building_df.index, building_df["ElImp"])
plt.ylabel("Imported electricity use, ElImp [Wh/h]")
plt.title("Imported electricity for building in file")
plt.tight_layout()
plt.show()

# ------------------------------------------------------------------
# 2) Classify the building category and heating type
# ------------------------------------------------------------------
# This section predicts class probabilities for building type and heating type of the electric load profile.
# It then plots the probability distribution to show the most likely classes.
#
# Class label explanation:
# Apb = apartment block, Apt = apartment, Cab = cabin, Hou = house,
# Hsp = hospital, Htl = hotel, Kdg = kindergarten, Nsh = nursing home,
# Off = office, Sch = school, Shp = shop, Uni = university.
# NEH = non-electric heating, EH = electric heating.
# ------------------------------------------------------------------

class_prob = CofClassifyClassify.predict_class_probabilities_ts(building_df)
class_prob_fig = CofClassifyClassify.plot_building_class_probabilities(class_prob)

# ------------------------------------------------------------------
# 3) Disaggregate electricity for heating
# ------------------------------------------------------------------
# This section estimates the amount of electricity use that is used
# for heating, based on the input time series (building_df).
#
# disaggregate_trea_ts(ts) is the simplest entry point:
# - extracts features internally
# - loads a default model if none is provided (if you implemented that)
# - returns predicted heating electricity time series
# Alternatively, extract features first, then estimate disaggregation. 
#
# Then we plot "total vs estimated heating".
# ------------------------------------------------------------------

#disaggregation_preds = CofDisaggregationDisaggregate.disaggregate_trea_ts(building_df)
disaggregation_features = CofDisaggregationFeatureExtraction.get_extracted_features_from_building_ts(building_df) 
disaggregation_preds = CofDisaggregationDisaggregate.disaggregate_feature_extracted_building(disaggregation_features)

disaggregation_fig = CofDisaggregationDisaggregate.plot_disaggregation(
    building_df["ElImp"],
    disaggregation_preds,
    bname="Building",
)





