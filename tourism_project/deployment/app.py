"""Streamlit interface for the Tourism pre-contact purchase propensity model."""
from pathlib import Path
import json
import joblib
import pandas as pd
import streamlit as st

HERE = Path(__file__).resolve().parent
st.set_page_config(page_title="Visit with Us | Wellness Tourism", page_icon="🌿", layout="wide")
st.title("Visit with Us")
st.subheader("Wellness Tourism · Customer purchase propensity")
st.caption("A decision-support estimate using customer information available before sales contact.")

@st.cache_resource
def load_model(model_mtime, schema_mtime):
    # mtime keys invalidate the cache when deployment artifacts are updated.
    return joblib.load(HERE / "model.joblib"), json.loads((HERE / "schema.json").read_text())

if not (HERE / "model.joblib").exists() or not (HERE / "schema.json").exists():
    st.warning("The model has not been trained yet. Complete the GitHub Actions pipeline first.")
    st.stop()
model, schema = load_model((HERE / "model.joblib").stat().st_mtime_ns,
                           (HERE / "schema.json").stat().st_mtime_ns)
labels = {"Age": "Age (years)", "CityTier": "City tier", "NumberOfPersonVisiting": "People visiting",
          "PreferredPropertyStar": "Preferred hotel stars", "NumberOfTrips": "Annual trips",
          "Passport": "Passport (0: No, 1: Yes)", "OwnCar": "Own car (0: No, 1: Yes)",
          "NumberOfChildrenVisiting": "Children below age 5 visiting", "MonthlyIncome": "Monthly income",
          "Occupation": "Occupation", "Gender": "Gender", "MaritalStatus": "Marital status",
          "Designation": "Designation"}
with st.form("tourism_customer"):
    left, right = st.columns(2)
    inputs = {}
    for index, (col, spec) in enumerate(schema["numeric"].items()):
        with (left if index % 2 == 0 else right):
            inputs[col] = st.number_input(labels[col], min_value=float(spec["min"]),
                                         max_value=float(spec["max"]), value=float(spec["default"]),
                                         step=1.0 if col != "MonthlyIncome" else 100.0)
    for index, (col, choices) in enumerate(schema["categorical"].items()):
        with (left if index % 2 == 0 else right):
            inputs[col] = st.selectbox(labels[col], choices)
    submitted = st.form_submit_button("Predict purchase propensity", type="primary")
if submitted:
    count_fields = ["CityTier", "NumberOfPersonVisiting", "PreferredPropertyStar", "NumberOfTrips",
                    "Passport", "OwnCar", "NumberOfChildrenVisiting"]
    if any(not float(inputs[col]).is_integer() for col in count_fields):
        st.error("City tier, ratings, binary flags and trip/visitor counts must be whole numbers.")
    elif inputs["NumberOfChildrenVisiting"] > inputs["NumberOfPersonVisiting"]:
        st.error("Children visiting cannot exceed the total number of people visiting.")
    else:
        frame = pd.DataFrame([inputs], columns=schema["features"])
        probability = float(model.predict_proba(frame)[0, 1])
        st.metric("Estimated purchase propensity", f"{probability:.1%}")
        if probability >= schema["threshold"]:
            st.success("Predicted class: Purchase (1)")
        else:
            st.info("Predicted class: No purchase (0)")
        st.caption(f"Classification threshold: {schema['threshold']:.2f}. This score is not a guarantee or a calibrated probability.")
st.divider()
st.caption("The model uses historical package purchases as a proxy for Wellness Tourism demand. "
           "It excludes pitch/follow-up attributes. Sensitive attributes require fairness review before real marketing use. "
           "Customer inputs are processed for this request and are not deliberately stored by this app.")
