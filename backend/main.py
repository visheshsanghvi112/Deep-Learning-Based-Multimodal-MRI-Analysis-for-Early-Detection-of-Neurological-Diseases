from pathlib import Path
from typing import List, Optional

import numpy as np
import pandas as pd
from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel


BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data" / "processed"
RESULTS_DIR = BASE_DIR / "results"

FEATURES_FILE = DATA_DIR / "oasis_complete_features_full.csv"
METRICS_FILE = RESULTS_DIR / "evaluation" / "evaluation_metrics.json"
PREDICTIONS_FILE = RESULTS_DIR / "evaluation" / "predictions.npz"


class DatasetSummary(BaseModel):
  dataset: str
  n_subjects: int
  n_cdr_labeled: int
  modality: str = "T1-weighted structural MRI"


class Metric(BaseModel):
  name: str
  value: float
  description: Optional[str] = None


class MetricsResponse(BaseModel):
  variant: str  # "baseline" | "prototype"
  dataset: str
  metrics: List[Metric]


class SubjectSummary(BaseModel):
  subject_id: str
  age: Optional[float]
  gender: Optional[str]
  cdr: Optional[float]
  mmse: Optional[float]
  nwbv: Optional[float]
  etiv: Optional[float]


class PaginatedSubjects(BaseModel):
  items: List[SubjectSummary]
  total: int
  page: int
  page_size: int


class SubjectPrediction(BaseModel):
  subject_id: str
  cdr_pred: Optional[float]
  mmse_pred: Optional[float]
  binary_prob: Optional[float]


app = FastAPI(
  title="NeuroScope API",
  description=(
    "Read-only API exposing OASIS-1 dataset summaries, metrics, and "
    "baseline predictions for the NeuroScope research portal."
  ),
)

app.add_middleware(
  CORSMiddleware,
  allow_origins=["*"],
  allow_credentials=False,
  allow_methods=["*"],
  allow_headers=["*"],
)

# Serve static interpretability figures
figures_dir = RESULTS_DIR / "figures"
if figures_dir.exists():
  app.mount("/static", StaticFiles(directory=str(figures_dir)), name="static")


def _load_features() -> pd.DataFrame:
  if not FEATURES_FILE.exists():
    raise RuntimeError(f"Features file not found: {FEATURES_FILE}")
  df = pd.read_csv(FEATURES_FILE)
  return df


def _load_metrics() -> dict:
  if not METRICS_FILE.exists():
    raise RuntimeError(f"Metrics file not found: {METRICS_FILE}")
  import json

  with METRICS_FILE.open("r", encoding="utf-8") as f:
    return json.load(f)


def _load_predictions() -> dict:
  if not PREDICTIONS_FILE.exists():
    raise RuntimeError(f"Predictions file not found: {PREDICTIONS_FILE}")
  data = np.load(PREDICTIONS_FILE, allow_pickle=True)
  return {k: data[k] for k in data.files}


@app.get("/api/oasis1/summary", response_model=DatasetSummary)
def get_oasis_summary() -> DatasetSummary:
  df = _load_features()
  n_subjects = len(df)
  n_cdr_labeled = int(df["CDR"].notna().sum()) if "CDR" in df.columns else 0
  return DatasetSummary(
    dataset="OASIS-1",
    n_subjects=n_subjects,
    n_cdr_labeled=n_cdr_labeled,
  )


@app.get("/api/oasis1/baseline", response_model=MetricsResponse)
def get_baseline_metrics() -> MetricsResponse:
  metrics_raw = _load_metrics()
  # Expecting baseline metrics to be stored under a "baseline" key; if not, fall back to top-level.
  baseline = metrics_raw.get("baseline", metrics_raw)
  metric_list = []
  for name, value in baseline.items():
    if isinstance(value, (int, float)):
      metric_list.append(Metric(name=name, value=float(value)))
  return MetricsResponse(
    variant="baseline",
    dataset="OASIS-1",
    metrics=metric_list,
  )


@app.get("/api/oasis1/prototype", response_model=MetricsResponse)
def get_prototype_metrics() -> MetricsResponse:
  metrics_raw = _load_metrics()
  proto = metrics_raw.get("prototype", {})
  metric_list = []
  for name, value in proto.items():
    if isinstance(value, (int, float)):
      metric_list.append(Metric(name=name, value=float(value)))
  return MetricsResponse(
    variant="prototype",
    dataset="OASIS-1",
    metrics=metric_list,
  )


@app.get("/api/oasis1/subjects", response_model=PaginatedSubjects)
def list_subjects(
  page: int = Query(1, ge=1),
  page_size: int = Query(20, ge=1, le=100),
) -> PaginatedSubjects:
  df = _load_features()
  total = len(df)
  start = (page - 1) * page_size
  end = start + page_size
  df_page = df.iloc[start:end]

  items: List[SubjectSummary] = []
  for _, row in df_page.iterrows():
    items.append(
      SubjectSummary(
        subject_id=str(row.get("SUBJECT_ID", "")),
        age=float(row["AGE"]) if pd.notna(row.get("AGE")) else None,
        gender=str(row.get("GENDER")) if pd.notna(row.get("GENDER")) else None,
        cdr=float(row["CDR"]) if pd.notna(row.get("CDR")) else None,
        mmse=float(row["MMSE"]) if pd.notna(row.get("MMSE")) else None,
        nwbv=float(row["NWBV"]) if pd.notna(row.get("NWBV")) else None,
        etiv=float(row["eTIV"]) if pd.notna(row.get("eTIV")) else None,
      )
    )

  return PaginatedSubjects(
    items=items,
    total=total,
    page=page,
    page_size=page_size,
  )


@app.get("/api/oasis1/subjects/{subject_id}/baseline", response_model=SubjectPrediction)
def get_subject_baseline(subject_id: str) -> SubjectPrediction:
  preds = _load_predictions()
  ids = preds.get("subject_ids")
  if ids is None:
    raise HTTPException(status_code=500, detail="subject_ids not found in predictions file")

  # subject_ids may be bytes or strings depending on how saved
  ids_list = [sid.decode("utf-8") if isinstance(sid, (bytes, bytearray)) else str(sid) for sid in ids]
  try:
    idx = ids_list.index(subject_id)
  except ValueError:
    raise HTTPException(status_code=404, detail="Subject not found in predictions")

  def _safe_get(name: str) -> Optional[float]:
    arr = preds.get(name)
    if arr is None or not len(arr) > idx:
      return None
    val = arr[idx]
    try:
      return float(val)
    except (TypeError, ValueError):
      return None

  return SubjectPrediction(
    subject_id=subject_id,
    cdr_pred=_safe_get("cdr_pred"),
    mmse_pred=_safe_get("mmse_pred"),
    binary_prob=_safe_get("binary_prob"),
  )


# For local dev:
# uvicorn backend.main:app --reload --port 8000


