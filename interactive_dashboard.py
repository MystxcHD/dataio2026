import streamlit as st
import pandas as pd
import numpy as np
import joblib

# Load trained model
gb_model = joblib.load("housing_model.pkl")   # save this from your main script

st.title("NYC Housing Price Predictor")

# Interactive inputs
sqft = st.slider("Building Area (sqft)", 200, 10000, 2500)
year = st.slider("Year Built", 1850, 2025, 1940)
borough = st.selectbox("Borough", ["Manhattan", "Bronx", "Brooklyn", "Queens", "Staten Island"])

boro_map = {"Manhattan":1, "Bronx":2, "Brooklyn":3, "Queens":4, "Staten Island":5}
boro_code = boro_map[borough]

# Build input row
input_data = pd.DataFrame(columns=gb_model.feature_names_in_)
input_data.loc[0] = 0
input_data["bldgarea"] = sqft
input_data["yearbuilt"] = year

col_name = f"borough_x_{boro_code}"
if col_name in input_data.columns:
    input_data[col_name] = 1

# Predict
pred = gb_model.predict(input_data)[0]

st.metric("Estimated Price", f"${pred:,.0f}")