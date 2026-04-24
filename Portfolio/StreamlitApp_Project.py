import os, sys, warnings, tarfile, tempfile
import numpy as np
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt
import posixpath
import json

import joblib
import boto3
import sagemaker
from sagemaker.predictor import Predictor
from sagemaker.serializers import NumpySerializer
from sagemaker.deserializers import NumpyDeserializer

from sklearn.pipeline import Pipeline
import shap

from joblib import dump, load

# ── Setup ──────────────────────────────────────────────────────────────────────
warnings.simplefilter("ignore")

current_dir  = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.abspath(os.path.join(current_dir, '..'))
if project_root not in sys.path:
    sys.path.append(project_root)

# ── Model config ───────────────────────────────────────────────────────────────
MODEL_INFO = {
    "endpoint"  : "fraud-detection-endpoint-1",
    "explainer" : "explainer_fraud.shap",
    "pipeline"  : "finalized_loan_model.tar.gz",
    # Feature names and metadata for the UI form
    "inputs": [
        {"name": "TransactionAmt",     "type": "number", "min": 0.0,   "max": 20000.0, "default": 100.0,  "step": 1.0,  "label": "Transaction Amount ($)"},
        {"name": "TransactionAmt_log", "type": "number", "min": 0.0,   "max": 10.0,    "default": 4.6,    "step": 0.01, "label": "Log(Transaction Amount)"},
        {"name": "card1",              "type": "number", "min": 1000.0, "max": 20000.0, "default": 9500.0, "step": 1.0,  "label": "Card 1 ID"},
        {"name": "card2",              "type": "number", "min": 100.0,  "max": 600.0,   "default": 321.0,  "step": 1.0,  "label": "Card 2 Code"},
        {"name": "addr1",              "type": "number", "min": 100.0,  "max": 540.0,   "default": 299.0,  "step": 1.0,  "label": "Billing Address Code"},
        {"name": "C1",                 "type": "number", "min": 0.0,   "max": 4000.0,  "default": 1.0,    "step": 1.0,  "label": "C1 (Card Count Feature)"},
        {"name": "C2",                 "type": "number", "min": 0.0,   "max": 2000.0,  "default": 1.0,    "step": 1.0,  "label": "C2 (Card Count Feature)"},
        {"name": "C14",                "type": "number", "min": 0.0,   "max": 1000.0,  "default": 0.0,    "step": 1.0,  "label": "C14 (Count Feature)"},
        {"name": "hour_of_day",        "type": "number", "min": 0.0,   "max": 23.0,    "default": 12.0,   "step": 1.0,  "label": "Hour of Day (0–23)"},
        {"name": "day_of_week",        "type": "number", "min": 0.0,   "max": 6.0,     "default": 2.0,    "step": 1.0,  "label": "Day of Week (0=Mon, 6=Sun)"},
    ],
    "keys": ["TransactionAmt", "TransactionAmt_log", "card1", "card2", "addr1",
             "C1", "C2", "C14", "hour_of_day", "day_of_week"],
}

# ── AWS credentials from Streamlit secrets ─────────────────────────────────────
aws_id      = st.secrets["aws_credentials"]["AWS_ACCESS_KEY_ID"]
aws_secret  = st.secrets["aws_credentials"]["AWS_SECRET_ACCESS_KEY"]
aws_token   = st.secrets["aws_credentials"]["AWS_SESSION_TOKEN"]
aws_bucket  = st.secrets["aws_credentials"]["AWS_BUCKET"]

# ── Session management ─────────────────────────────────────────────────────────
@st.cache_resource
def get_session(aws_id, aws_secret, aws_token):
    return boto3.Session(
        aws_access_key_id=aws_id,
        aws_secret_access_key=aws_secret,
        aws_session_token=aws_token,
        region_name='us-east-1'
    )

session    = get_session(aws_id, aws_secret, aws_token)
sm_session = sagemaker.Session(boto_session=session)

# ── Load helpers ───────────────────────────────────────────────────────────────
def load_pipeline(_session, bucket, key):
    s3_client = _session.client('s3')
    filename  = MODEL_INFO["pipeline"]
    s3_client.download_file(Filename=filename, Bucket=bucket, Key=f"{key}/{os.path.basename(filename)}")
    with tarfile.open(filename, "r:gz") as tar:
        tar.extractall(path=".")
        joblib_file = [f for f in tar.getnames() if f.endswith('.joblib')][0]
    return joblib.load(joblib_file)

def load_shap_explainer(_session, bucket, key, local_path):
    s3_client = _session.client('s3')
    if not os.path.exists(local_path):
        s3_client.download_file(Filename=local_path, Bucket=bucket, Key=key)
    with open(local_path, "rb") as f:
        return load(f)

# ── Prediction via SageMaker endpoint ─────────────────────────────────────────
def call_model_api(input_df):
    predictor = Predictor(
        endpoint_name=MODEL_INFO["endpoint"],
        sagemaker_session=sm_session,
        serializer=NumpySerializer(),
        deserializer=NumpyDeserializer()
    )
    try:
        # Load feature names from S3
        s3_client = session.client('s3')
        s3_client.download_file(Bucket=aws_bucket, Key='feature_names.json', Filename='feature_names.json')
        with open('feature_names.json') as f:
            feature_names = json.load(f)

        # Create full row with zeros, fill in user inputs
        full_row = pd.DataFrame([np.zeros(len(feature_names))], columns=feature_names)
        for k, v in input_df.items():
            if k in full_row.columns:
                full_row[k] = v

        raw_pred = predictor.predict(full_row.values)
        pred_val = int(pd.DataFrame(raw_pred).values[-1][0])
        best_pipeline = load_pipeline(session, aws_bucket, 'sklearn-pipeline-deployment')
        proba = best_pipeline.predict_proba(full_row)[0][1]
        return pred_val, round(float(proba), 4), 200, full_row
    except Exception as e:
        return None, None, f"Error: {str(e)}", None

# ── SHAP explanation display ───────────────────────────────────────────────────
def display_explanation(input_df):
    explainer_name = MODEL_INFO["explainer"]
    local_path     = os.path.join(tempfile.gettempdir(), explainer_name)

    explainer    = load_shap_explainer(session, aws_bucket,
                       posixpath.join('explainer', explainer_name), local_path)
    best_pipeline = load_pipeline(session, aws_bucket, 'sklearn-pipeline-deployment')

    # Run preprocessing (all steps except final model)
    preprocessing_pipeline = Pipeline(steps=best_pipeline.steps[:-1])
    input_transformed = preprocessing_pipeline.transform(input_df)
    feature_names_out = MODEL_INFO["keys"]  # simplified; adapt if pipeline adds features
    input_df_tf = pd.DataFrame(input_transformed, columns=feature_names_out)

    shap_values = explainer(input_df_tf)

    st.subheader("🔍 Decision Transparency (SHAP Waterfall)")
    fig, ax = plt.subplots(figsize=(10, 5))
    shap.plots.waterfall(shap_values[0], max_display=10, show=False)
    st.pyplot(fig)

    top_feature = (pd.Series(shap_values[0].values, index=shap_values[0].feature_names)
                   .abs().idxmax())
    st.info(f"**Business Insight:** The most influential factor in this prediction was **{top_feature}**.")

# ── Streamlit UI ───────────────────────────────────────────────────────────────
st.set_page_config(page_title="Fraud Detection — ML App", layout="wide")

st.title("🔐 Fraud Detection — Real-Time Transaction Scoring")
st.markdown("""
This app uses a **LightGBM classifier** trained on the IEEE-CIS Fraud Detection dataset.  
Enter transaction details below to receive an instant fraud risk score and a SHAP explanation of the decision.
""")

st.divider()

with st.form("fraud_form"):
    st.subheader("📋 Transaction Details")
    cols = st.columns(2)
    user_inputs = {}

    for i, inp in enumerate(MODEL_INFO["inputs"]):
        with cols[i % 2]:
            user_inputs[inp['name']] = st.number_input(
                label=inp['label'],
                min_value=float(inp['min']),
                max_value=float(inp['max']),
                value=float(inp['default']),
                step=float(inp['step'])
            )

    submitted = st.form_submit_button("🔍 Score Transaction", use_container_width=True)

if submitted:
    data_row = [user_inputs[k] for k in MODEL_INFO["keys"]]
    input_df = pd.DataFrame([data_row], columns=MODEL_INFO["keys"])

    with st.spinner("Running prediction..."):
        pred_val, proba, status, full_row = call_model_api(input_df)

    if status == 200:
        st.divider()
        col1, col2, col3 = st.columns(3)

        with col1:
            label = "🚨 FRAUDULENT" if pred_val == 1 else "✅ LEGITIMATE"
            st.metric("Prediction", label)

        with col2:
            st.metric("Fraud Probability", f"{proba:.1%}")

        with col3:
            risk = "HIGH" if proba > 0.7 else ("MEDIUM" if proba > 0.3 else "LOW")
            color = "🔴" if risk == "HIGH" else ("🟡" if risk == "MEDIUM" else "🟢")
            st.metric("Risk Level", f"{color} {risk}")

        # Fraud probability gauge bar
        st.progress(proba, text=f"Fraud Score: {proba:.1%}")

        if pred_val == 1:
            st.error("⚠️ This transaction has been flagged as potentially fraudulent. Recommended action: **Manual review or decline.**")
        else:
            st.success("✅ This transaction appears legitimate. Recommended action: **Approve.**")

        st.divider()
        with st.expander("📊 View SHAP Explanation", expanded=True):
            display_explanation(full_row)

    else:
        st.error(status)

# ── Sidebar info ───────────────────────────────────────────────────────────────
with st.sidebar:
    st.header("ℹ️ About")
    st.markdown("""
**Model:** LightGBM Classifier  
**Dataset:** IEEE-CIS Fraud Detection  
**Resampling:** SMOTE  
**Primary Metric:** ROC-AUC  

---
**Interpretation Guide:**
- 🟢 **Low risk** (< 30%): Approve
- 🟡 **Medium risk** (30–70%): Review
- 🔴 **High risk** (> 70%): Decline / Investigate

---
**Top fraud signals:**
- Unusual transaction timing
- Mismatched card & address codes
- High C1/C2 count features
- Large transaction amounts
    """)
    st.caption("Ava Annino — ML Project | Spring 2026")
