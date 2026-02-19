import os
import sys
import warnings
import numpy as np
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt
import posixpath
import joblib
import tarfile
import tempfile
import boto3
import sagemaker
from sagemaker.predictor import Predictor
from sagemaker.serializers import NumpySerializer
from sagemaker.deserializers import NumpyDeserializer
import shap

warnings.simplefilter("ignore")

# -----------------------------
# AWS + PATH SETUP
# -----------------------------

current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.abspath(os.path.join(current_dir, ".."))
if project_root not in sys.path:
    sys.path.append(project_root)

aws_id = st.secrets["aws_credentials"]["AWS_ACCESS_KEY_ID"]
aws_secret = st.secrets["aws_credentials"]["AWS_SECRET_ACCESS_KEY"]
aws_token = st.secrets["aws_credentials"]["AWS_SESSION_TOKEN"]
aws_bucket = st.secrets["aws_credentials"]["AWS_BUCKET"]
aws_endpoint = st.secrets["aws_credentials"]["AWS_ENDPOINT"]

@st.cache_resource
def get_session(aws_id, aws_secret, aws_token):
    return boto3.Session(
        aws_access_key_id=aws_id,
        aws_secret_access_key=aws_secret,
        aws_session_token=aws_token,
        region_name="us-east-1",
    )

session = get_session(aws_id, aws_secret, aws_token)
sm_session = sagemaker.Session(boto_session=session)

# -----------------------------
# MODEL CONFIG
# -----------------------------

MODEL_INFO = {
    "endpoint": aws_endpoint,
    "explainer": "explainer.shap",
    "pipeline": "finalized_model.tar.gz",
    "keys": [
        "Momentum_5",
        "Volatility_10",
        "High_Low_Spread",
        "MA_Crossover"
    ],
    "inputs": [
        {"name": "Momentum_5", "min": -1.0, "max": 1.0, "default": 0.0, "step": 0.01},
        {"name": "Volatility_10", "min": 0.0, "max": 1.0, "default": 0.05, "step": 0.01},
        {"name": "High_Low_Spread", "min": 0.0, "max": 1.0, "default": 0.02, "step": 0.01},
        {"name": "MA_Crossover", "min": -1.0, "max": 1.0, "default": 0.0, "step": 1.0},
    ],
}

# -----------------------------
# SHAP LOADING
# -----------------------------

def load_shap_explainer(_session, bucket, key, local_path):
    s3_client = _session.client("s3")

    if not os.path.exists(local_path):
        s3_client.download_file(bucket, key, local_path)

    with open(local_path, "rb") as f:
        return shap.Explainer.load(f)

# -----------------------------
# CALL SAGEMAKER
# -----------------------------

def call_model_api(input_df):
    predictor = Predictor(
        endpoint_name=MODEL_INFO["endpoint"],
        sagemaker_session=sm_session,
        serializer=NumpySerializer(),
        deserializer=NumpyDeserializer(),
    )

    try:
        raw_pred = predictor.predict(input_df.values.astype(np.float32))
        pred_val = float(raw_pred[0][0])
        return round(pred_val, 4), 200
    except Exception as e:
        return f"Error: {str(e)}", 500

# -----------------------------
# SHAP DISPLAY
# -----------------------------

def display_explanation(input_df):
    explainer_name = MODEL_INFO["explainer"]

    explainer = load_shap_explainer(
        session,
        aws_bucket,
        posixpath.join("explainer", explainer_name),
        os.path.join(tempfile.gettempdir(), explainer_name),
    )

    shap_values = explainer(input_df)

    fig = plt.figure()
    shap.plots.waterfall(shap_values[0], max_display=10, show=False)
    st.pyplot(fig)

# -----------------------------
# UI
# -----------------------------

st.set_page_config(page_title="NVDA ML Deployment", layout="wide")
st.title("NVDA Model Deployment")

with st.form("prediction_form"):

    cols = st.columns(2)
    user_inputs = {}

    for i, inp in enumerate(MODEL_INFO["inputs"]):
        with cols[i % 2]:
            user_inputs[inp["name"]] = st.number_input(
                inp["name"].replace("_", " "),
                min_value=inp["min"],
                max_value=inp["max"],
                value=inp["default"],
                step=inp["step"],
            )

    submitted = st.form_submit_button("Run Prediction")

if submitted:

    data_row = [user_inputs[k] for k in MODEL_INFO["keys"]]

    input_df = pd.DataFrame([data_row], columns=MODEL_INFO["keys"])

    result, status = call_model_api(input_df)

    if status == 200:
        st.metric("Predicted NVDA Future Return", result)
        display_explanation(input_df)
    else:
        st.error(result)
