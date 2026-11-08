"""Prepare leakage-aware tourism features and reproducible train/test splits."""
from pathlib import Path
import json
import pandas as pd
from sklearn.model_selection import train_test_split
from data_register import register

ROOT = Path(__file__).resolve().parents[2]
NUMERIC_FEATURES = ["Age", "CityTier", "NumberOfPersonVisiting", "PreferredPropertyStar",
                    "NumberOfTrips", "Passport", "OwnCar", "NumberOfChildrenVisiting", "MonthlyIncome"]
CATEGORICAL_FEATURES = ["Occupation", "Gender", "MaritalStatus", "Designation"]
FEATURES = NUMERIC_FEATURES + CATEGORICAL_FEATURES
# Interaction attributes are unavailable before contact. Historical purchase is the
# supplied label; it is not direct proof of uptake for a newly launched package.
EXCLUDED_FEATURES = ["CustomerID", "TypeofContact", "DurationOfPitch", "NumberOfFollowups",
                     "ProductPitched", "PitchSatisfactionScore"]

def prepare():
    register()
    df = pd.read_csv(ROOT / "tourism_project/data/tourism.csv")
    df = df.loc[:, ~df.columns.str.startswith("Unnamed:")].copy()
    for col in CATEGORICAL_FEATURES:
        df[col] = df[col].str.strip()
    df["Gender"] = df["Gender"].replace({"Fe Male": "Female"})
    df["Occupation"] = df["Occupation"].replace({"Free Lancer": "Freelancer"})
    # Treat Single and Unmarried as the same category for this tourism use case.
    df["MaritalStatus"] = df["MaritalStatus"].replace({"Unmarried": "Single"})
    for col in NUMERIC_FEATURES:
        df[col] = pd.to_numeric(df[col], errors="raise")
    X = df[FEATURES]
    y = df["ProdTaken"].astype(int)
    Xtrain, Xtest, ytrain, ytest = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y)
    for name, values in {"Xtrain": Xtrain, "Xtest": Xtest, "ytrain": ytrain, "ytest": ytest}.items():
        values.to_csv(ROOT / f"{name}.csv", index=False)
    report = {"train_rows": len(Xtrain), "test_rows": len(Xtest), "random_state": 42,
              "test_size": 0.2, "features": FEATURES, "excluded_features": EXCLUDED_FEATURES,
              "train_purchase_rate": float(ytrain.mean()), "test_purchase_rate": float(ytest.mean())}
    (ROOT / "tourism_project/reports/preparation.json").write_text(json.dumps(report, indent=2))
    print(json.dumps(report, indent=2))

if __name__ == "__main__":
    prepare()
