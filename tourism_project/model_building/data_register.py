"""Validate and register the repository's tourism dataset."""
from pathlib import Path
import hashlib
import json
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
CSV_LOCAL = ROOT / "tourism_project/data/tourism.csv"
EXPECTED_COLUMNS = [
    "CustomerID", "ProdTaken", "Age", "TypeofContact", "CityTier", "DurationOfPitch",
    "Occupation", "Gender", "NumberOfPersonVisiting", "NumberOfFollowups",
    "ProductPitched", "PreferredPropertyStar", "MaritalStatus", "NumberOfTrips",
    "Passport", "PitchSatisfactionScore", "OwnCar", "NumberOfChildrenVisiting",
    "Designation", "MonthlyIncome",
]

def register():
    if not CSV_LOCAL.is_file():
        raise FileNotFoundError(f"Upload the assignment dataset to {CSV_LOCAL}")
    df = pd.read_csv(CSV_LOCAL)
    missing = sorted(set(EXPECTED_COLUMNS) - set(df.columns))
    if missing:
        raise ValueError(f"Missing expected columns: {missing}")
    if df.empty or df["ProdTaken"].isna().any() or set(df["ProdTaken"].unique()) != {0, 1}:
        raise ValueError("ProdTaken must contain both classes 0 and 1, with no missing labels.")
    if df["CustomerID"].isna().any() or df["CustomerID"].duplicated().any():
        raise ValueError("CustomerID must be nonmissing and unique; investigate before splitting.")
    summary = {
        "rows": len(df), "columns": list(df.columns),
        "target_counts": {str(k): int(v) for k, v in df["ProdTaken"].value_counts().items()},
        "missing_counts": {str(k): int(v) for k, v in df.isna().sum().items()},
        "duplicate_rows": int(df.duplicated().sum()),
        "sha256": hashlib.sha256(CSV_LOCAL.read_bytes()).hexdigest(),
    }
    out = ROOT / "tourism_project/reports"
    out.mkdir(parents=True, exist_ok=True)
    (out / "data_summary.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2))
    return summary

if __name__ == "__main__":
    register()
