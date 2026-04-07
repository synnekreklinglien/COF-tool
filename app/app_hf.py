import sys
from pathlib import Path
import traceback
import io
import tempfile

import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import requests

st.set_page_config(page_title="COF-tool")

# -----------------------------
# HF backend URL
# -----------------------------
HF_BACKEND_URL = "https://synnekreklinglien-cof-tool.hf.space"
DISAGGREGATE_ENDPOINT = f"{HF_BACKEND_URL}/disaggregate"


# -----------------------------
# Page header / intro
# -----------------------------
st.title("COF-Tool: A tool for classification and heat load disaggregation of hourly electricity load profiles from buildings")

st.markdown(
    """
This tool classifies building characteristics and estimates electricity use for heating from the hourly electric load profile of a Norwegian building. It consists of two modules: a classification module and a disaggregation module.

The classification module analyses the building's hourly electricity load profile and outdoor temperature to estimate the most likely building category and heating system type.

The disaggregation module estimates the electricity use for heating as an hourly time series. It is developed for larger buildings with electric heating, such as apartment blocks and non-residential buildings, and has not been trained or validated for smaller residential units such as single-family houses or individual apartments.

To use the tool, upload a building data file below containing hourly electricity use and corresponding hourly outdoor temperature. See the example file for the required format. The models are trained on historical data from Norwegian buildings and are intended only for use on Norwegian building data.

The tool and its outputs may be used for any purpose under the MIT Licence (2026); any use of results from the tool must provide credit as stated."""
)

with st.expander("Acknowledgement and references"):
    st.markdown(
        """
This tool was developed as part of the PhD project of Synne Krekling Lien and the research project Coincidence factors and peak loads of buildings in the Norwegian low carbon society (COFACTOR).
The authors gratefully acknowledge support from the Research Council of Norway (project number 326891), as well as contributions from research partners, industry partners, and data providers.

The source code for this tool is available on GitHub: https://github.com/synnekreklinglien/COF-tool

If you use this tool or any results produced by it, you must cite the following article:
ARTICLE REFERENCE

Contact: Synne.lien@sintef.no

Last updated: 12.03.2026
"""
    )

st.divider()


# -----------------------------
# Make project root importable
# -----------------------------
project_root = Path(__file__).resolve().parents[1]
if not (project_root / "modules").exists():
    st.error(f"project_root does not contain modules/: {project_root}")
    st.stop()

modules_dir = project_root / "modules"
if str(modules_dir) not in sys.path:
    sys.path.insert(0, str(modules_dir))

sys.path.insert(0, str(project_root))

import modules.CofClassification.lib.ReadtreaData as treaData
import modules.CofClassification.lib.CofClassifyClassify as CofClassifyClassify
import modules.CofClassification.lib.CofClassifyFeatureExtraction as CofClassifyFeatureExtraction
import modules.CofDisaggregation.lib.CofDisaggregationDisaggregate as CofDisaggregationDisaggregate


# -----------------------------
# Helpers
# -----------------------------
def get_static_dir() -> Path:
    return Path(__file__).resolve().parent / "static"


def reset_state_on_new_upload(uploaded_name: str, cols: list) -> None:
    if ("uploaded_filename" not in st.session_state) or (st.session_state["uploaded_filename"] != uploaded_name):
        st.session_state["uploaded_filename"] = uploaded_name
        st.session_state["original_cols"] = cols
        for k in ["ran_analyses", "class_prob", "class_prob_fig", "df_out_disagg", "disagg_fig"]:
            st.session_state.pop(k, None)


def call_disaggregate_api(df: pd.DataFrame) -> pd.DataFrame:
    """
    Serialise df as parquet and POST to the processing backend.
    The index is named 'TimeStamp' before sending so that reset_index()
    inside the feature extraction modules produces the expected column name.
    Returns the response dataframe with ElHeat_est column added.
    """
    df_to_send = df.copy()
    df_to_send.index.name = "TimeStamp"

    buf = io.BytesIO()
    df_to_send.to_parquet(buf, index=True)
    buf.seek(0)

    try:
        response = requests.post(
            DISAGGREGATE_ENDPOINT,
            files={"file": ("building.parquet", buf, "application/octet-stream")},
            timeout=600,
        )
    except requests.exceptions.Timeout:
        raise RuntimeError(
            "The disaggregation request timed out. The processing server may be starting up "
            "or under load. Please wait a minute and try again."
        )

    if response.status_code != 200:
        detail = response.json().get("detail", response.text) if response.content else response.text
        raise RuntimeError(f"Processing returned status {response.status_code}: {detail}")

    df_out = pd.read_parquet(io.BytesIO(response.content))
    return df_out


# -----------------------------
# Upload + example file
# -----------------------------
st.subheader("Upload a building file here:")

uploaded = st.file_uploader(
    "Building file uploader. Download example file for correct format.",
    type=["txt"],
    accept_multiple_files=False,
)

with st.expander("Need an example file?"):
    static_dir = get_static_dir()
    file_path = static_dir / "Cof_example_file.txt"
    try:
        file_bytes = file_path.read_bytes()
        st.download_button(
            "Download example file",
            data=file_bytes,
            file_name="Cof_example_file.txt",
            mime="text/plain",
            key="download_example_file",
        )
    except FileNotFoundError:
        st.error("Example file is missing from the static folder.")

    st.markdown(
        """
For more example files and information about the file format and allowed columns, see:
- [COFACTOR Drammen dataset - 4 years of hourly energy use data from 45 public buildings in Drammen, Norway](https://www.nature.com/articles/s41597-025-04708-3)
- [COFACTOR-SBHUB Oslo: Hourly Sub-Metered Energy Use Data from 48 public School Buildings in Oslo, Norway](https://www.sciencedirect.com/science/article/pii/S2352340925010091)
"""
    )

st.divider()

# -----------------------------
# Main logic
# -----------------------------
if uploaded is None:
    st.info("Upload a .txt building file to get started.")
    st.stop()

# Write to a temporary file that is deleted automatically after reading
with tempfile.NamedTemporaryFile(suffix=".txt", delete=False) as tmp_file:
    tmp_path = Path(tmp_file.name)
    tmp_file.write(uploaded.getbuffer())

try:
    meta, df = treaData.getTreaCsv(str(tmp_path))
except Exception:
    tmp_path.unlink(missing_ok=True)
    st.error("The uploaded file could not be read. Please check that it is in the required format and does not contain illegal columns.")
    st.stop()
finally:
    tmp_path.unlink(missing_ok=True)

reset_state_on_new_upload(uploaded.name, df.columns.tolist())

st.subheader("Select analysis:")

show_preview = st.checkbox("Show data preview", value=True)
run_classification = st.checkbox("Run building classification")
run_disaggregation = st.checkbox("Run disaggregation")

if st.button("Run selected analyses", key="run_selected"):
    st.session_state["ran_analyses"] = True

if not st.session_state.get("ran_analyses", False):
    st.stop()

if not any([show_preview, run_classification, run_disaggregation]):
    st.info("Select at least one option above before running.")
    st.stop()

# -----------------------------
# 1) Data preview
# -----------------------------
if show_preview:
    st.markdown("### Data preview")
    st.markdown(
        """
This preview shows the first rows of the uploaded time series from the building file as well as a
plot of the full time series of the total electricity (ElImp or ElTot) or total energy (X) if present.
"""
    )

    st.write("Here are the first rows of the time series:")
    st.dataframe(df.head())

    # ElImp = total electricity import; ElTot / X = total energy
    plot_col = next((c for c in ["ElImp", "ElTot", "X"] if c in df.columns), None)

    if plot_col is not None:
        col_label = {
            "ElImp": "total electricity import (ElImp)",
            "ElTot": "total electricity use (ElTot)",
            "X": "total energy use (X)",
        }.get(plot_col, plot_col)
        st.write(f"Plot of {col_label} against the time series index:")
        fig_prev, ax = plt.subplots()
        ax.plot(df.index, df[plot_col])
        ax.set_xlabel("Time")
        ax.set_ylabel(f"{plot_col} [Wh/h]")
        ax.set_title(f"{plot_col} vs time")
        st.pyplot(fig_prev)
    else:
        st.info("No ElImp, ElTot, or X column found to plot in the preview.")

# -----------------------------
# 2) Classification
# -----------------------------
if run_classification:
    st.markdown("### Building classification")
    st.markdown(
        """
This module classifies the building based on its hourly electricity load profile and outdoor temperature. It estimates the most likely building category and whether the building has electric or non-electric heating.
The results are presented as probabilities across all building classes. A high probability for a given class indicates that the load profile is characteristic of that building type.
"""
    )

    if st.session_state.get("class_prob") is None:
        with st.spinner("Running building classification..."):
            try:
                class_prob = CofClassifyClassify.predict_class_probabilities_ts(df)
                class_prob_fig = CofClassifyClassify.plot_building_class_probabilities(class_prob)
            except Exception:
                st.error("Error during feature extraction or classification")
                st.code(traceback.format_exc())
                class_prob = None
                class_prob_fig = None

            st.session_state["class_prob"] = class_prob
            st.session_state["class_prob_fig"] = class_prob_fig

    class_prob = st.session_state.get("class_prob")
    class_prob_fig = st.session_state.get("class_prob_fig")

    if class_prob is not None:
        if isinstance(class_prob, pd.DataFrame):
            prob_series = class_prob.iloc[0]
        else:
            prob_series = class_prob

        try:
            sorted_probs = prob_series.sort_values(ascending=False)
            top_classes = sorted_probs.index.tolist()
            top_values = sorted_probs.values.tolist()
        except Exception:
            top_classes, top_values = [], []

        if len(top_values) >= 2:
            st.write(
                f"This building is most likely a {top_classes[0]} at {top_values[0]:.1%} "
                f"or a {top_classes[1]} at {top_values[1]:.1%}."
            )
        elif len(top_values) == 1:
            st.write(f"This building is most likely a {top_classes[0]} at {top_values[0]:.1%}.")
        else:
            st.write("Could not determine the most likely building types.")

        if class_prob_fig is not None:
            st.pyplot(class_prob_fig)

        st.write(
            "Apb = apartment block, Apt = apartment, Cab = cabin, Hou = house, "
            "Hsp = hospital, Htl = hotel, Kdg = kindergarten, Nsh = nursing home, "
            "Off = office, Sch = school, Shp = shop, Uni = university. "
            "NEH = non-electric heating, EH = electric heating."
        )

        st.write("All class probabilities:")
        st.dataframe(class_prob)

# -----------------------------
# 3) Disaggregation
# -----------------------------
if run_disaggregation:
    st.markdown("### Disaggregation")

    st.write(
        """
This module estimates the electricity use for heating as an hourly time series, and separates it from other electricity use such as lighting, ventilation, and plug loads.

The model is a Categorical Boosting Regression (CatBoost) model trained on sub-metered data from 323 Norwegian buildings with district heating, using 288 features per hour.

For details on model development, validation, and limitations, see the reference in the acknowledgements section.
"""
    )

    if st.session_state.get("df_out_disagg") is None:
        with st.spinner("Running disaggregation (this may take a minute)..."):
            try:
                df_out = call_disaggregate_api(df)

                preds = df_out["ElHeat_est"]

                if "ElImp" not in df.columns:
                    raise KeyError(
                        "Column 'ElImp' not found in the uploaded file. "
                        "The disaggregation plot requires an ElImp column."
                    )

                disagg_fig = CofDisaggregationDisaggregate.plot_disaggregation(
                    df["ElImp"],
                    preds,
                    bname="Building",
                )

                # Keep only original uploaded columns, then add ElHeat_est aligned on index
                original_cols = st.session_state.get("original_cols", df.columns.tolist())
                df_result = df.loc[:, [c for c in original_cols if c in df.columns]].copy()
                df_result["ElHeat_est"] = preds.round(1)  # pandas aligns on index

                if df_result["ElHeat_est"].isna().any():
                    st.warning(
                        "Some rows in ElHeat_est are NaN after alignment. "
                        "The index returned from disaggregation may not fully match the uploaded data."
                    )

                st.session_state["df_out_disagg"] = df_result
                st.session_state["disagg_fig"] = disagg_fig

            except Exception as e:
                st.error(f"Disaggregation failed: {e}")
                st.code(traceback.format_exc())

    disagg_fig = st.session_state.get("disagg_fig")
    if disagg_fig is not None:
        st.pyplot(disagg_fig)

    df_out_disagg = st.session_state.get("df_out_disagg")
    if df_out_disagg is not None:
        st.warning(
            "**Note:** The disaggregation model is developed and validated for larger buildings with "
            "electric heating, such as apartment blocks and non-residential buildings. It has not been "
            "trained or validated for single-family houses or individual apartments, and results for "
            "such buildings should be interpreted with caution, even though the tool will still run for all building types.",
            icon="⚠️",
        )
        st.markdown("Download results:")
        st.caption("Download the full building time series including the estimated electricity use for heating, ElHeat_est.")
        st.download_button(
            label="Download results as TXT",
            data=df_out_disagg.to_csv(sep=";", index=True),
            file_name="building_data_with_heating_estimate.txt",
            mime="text/plain",
            key="download_disagg_txt",
        )