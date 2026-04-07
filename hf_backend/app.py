import io
import os
import sys
import shutil
from functools import lru_cache
from pathlib import Path

# Ensure /app is on sys.path so "modules" is importable as a top-level package
APP_DIR = Path(__file__).resolve().parent
if str(APP_DIR) not in sys.path:
    sys.path.insert(0, str(APP_DIR))

# Register aliases so joblib can unpickle ModelBundle, which was saved with
# CofClassification.lib.CofClassifyClassify as its module path
import modules.CofClassification.lib.CofClassifyClassify as _cc
sys.modules.setdefault("CofClassification", sys.modules.get("modules.CofClassification"))
sys.modules.setdefault("CofClassification.lib", sys.modules.get("modules.CofClassification.lib"))
sys.modules.setdefault("CofClassification.lib.CofClassifyClassify", _cc)

import pandas as pd
from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.responses import Response
from huggingface_hub import hf_hub_download

import modules.CofDisaggregation.lib.CofDisaggregationFeatureExtraction as FE
import modules.CofDisaggregation.lib.CofDisaggregationDisaggregate as DIS

MODEL_REPO = "synnekreklinglien/COF-tool-models"
DISAGG_MODEL_FILENAME = "disaggregation_model_All_AF_without_NEH.cbm"
CLASS_MODEL_FILENAME = "classification_model.joblib"

CLASS_MODEL_LOCAL_PATH = APP_DIR / "modules" / "CofClassification" / "resources" / "classification_model.joblib"

app = FastAPI()


def _ensure_classification_model():
    if not CLASS_MODEL_LOCAL_PATH.exists():
        print(f"Downloading {CLASS_MODEL_FILENAME} from {MODEL_REPO}...")
        downloaded_path = hf_hub_download(
            repo_id=MODEL_REPO,
            filename=CLASS_MODEL_FILENAME,
            token=os.environ.get("HF_TOKEN"),
        )
        CLASS_MODEL_LOCAL_PATH.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy(downloaded_path, CLASS_MODEL_LOCAL_PATH)
        print(f"Classification model placed at {CLASS_MODEL_LOCAL_PATH}")
    else:
        print(f"Classification model already present at {CLASS_MODEL_LOCAL_PATH}")


_ensure_classification_model()


@app.get("/")
def root():
    return {"status": "ok"}


@lru_cache(maxsize=1)
def load_disagg_model_cached():
    print(f"Downloading {DISAGG_MODEL_FILENAME} from {MODEL_REPO}...")
    downloaded_path = hf_hub_download(
        repo_id=MODEL_REPO,
        filename=DISAGG_MODEL_FILENAME,
        token=os.environ.get("HF_TOKEN"),
    )
    # load_disaggregation_model expects a directory and model name without extension
    model_path = Path(downloaded_path)
    return DIS.load_disaggregation_model(
        model_name=model_path.stem,
        savedir=model_path.parent,
    )


@app.post("/disaggregate")
async def disaggregate(file: UploadFile = File(...)):
    try:
        raw = await file.read()
        if not raw:
            raise HTTPException(status_code=400, detail="Empty upload")

        df = pd.read_parquet(io.BytesIO(raw))

        features = FE.get_extracted_features_from_building_ts(df)
        model = load_disagg_model_cached()
        preds = DIS.disaggregate_feature_extracted_building(features, model)

        df_out = df.copy()
        if isinstance(preds, pd.Series):
            df_out["ElHeat_est"] = preds.values
        else:
            df_out["ElHeat_est"] = preds

        buf = io.BytesIO()
        df_out.to_parquet(buf, index=True)
        buf.seek(0)

        return Response(
            content=buf.read(),
            media_type="application/octet-stream",
        )

    except HTTPException:
        raise
    except Exception as e:
        import traceback
        raise HTTPException(status_code=500, detail=traceback.format_exc())