import streamlit as st
import pandas as pd
import numpy as np
import joblib


# ============================================================
# PAGE CONFIGURATION
# ============================================================

st.set_page_config(
    page_title="Sugarcane Yield Predictor",
    page_icon="🌱",
    layout="centered"
)


# ============================================================
# LOAD MODEL
# ============================================================

@st.cache_resource
def load_model():

    model_data = joblib.load(
        "sugarcane_model.pkl"
    )

    return model_data


model_data = load_model()

model = model_data["model"]

scaler = model_data["scaler"]

features = model_data["features"]


# ============================================================
# HEADER
# ============================================================

st.title("🌱 Sugarcane Yield Predictor")

st.write(
    """
    Enter the agricultural and environmental conditions
    to predict the expected sugarcane yield.
    """
)

st.divider()


# ============================================================
# INPUT SECTION
# ============================================================

st.subheader("🌾 Farm Information")


col1, col2 = st.columns(2)


with col1:

    rainfall = st.number_input(
        "Rainfall (mm)",
        min_value=0.0,
        max_value=5000.0,
        value=2200.0,
        step=50.0
    )

    irrigation = st.number_input(
        "Irrigation (liters/ha)",
        min_value=0.0,
        max_value=10000.0,
        value=2100.0,
        step=50.0
    )

    fertilizer = st.number_input(
        "Fertilizer (kg/ha)",
        min_value=0.0,
        max_value=1000.0,
        value=130.0,
        step=5.0
    )


with col2:

    temperature = st.number_input(
        "Temperature (°C)",
        min_value=0.0,
        max_value=60.0,
        value=28.0,
        step=0.5
    )

    water_usage = st.number_input(
        "Water Usage (liters/ha)",
        min_value=0.0,
        max_value=10000.0,
        value=2150.0,
        step=50.0
    )

    soil_type = st.selectbox(
        "Soil Type",
        [
            "Clayey",
            "Loamy",
            "Sandy"
        ]
    )


st.divider()


# ============================================================
# INPUT PREPARATION FUNCTION
# ============================================================

def prepare_input(
    rainfall,
    irrigation,
    fertilizer,
    temperature,
    water_usage,
    soil_type
):

    # Start with numerical values
    input_data = {

        "Rainfall_mm": rainfall,

        "Irrigation_liters_ha": irrigation,

        "Fertilizer_kg_ha": fertilizer,

        "Temperature_C": temperature,

        "Water_Usage_liters_ha": water_usage
    }


    # Add soil columns
    input_data["Soil_Type_Clayey"] = (
        1 if soil_type == "Clayey" else 0
    )

    input_data["Soil_Type_Loamy"] = (
        1 if soil_type == "Loamy" else 0
    )

    input_data["Soil_Type_Sandy"] = (
        1 if soil_type == "Sandy" else 0
    )


    # Convert to DataFrame
    input_df = pd.DataFrame(
        [input_data]
    )


    # Make sure columns are exactly
    # the same as training data
    input_df = input_df.reindex(
        columns=features,
        fill_value=0
    )


    return input_df


# ============================================================
# PREDICTION FUNCTION
# ============================================================

def predict_sugarcane_yield(
    rainfall,
    irrigation,
    fertilizer,
    temperature,
    water_usage,
    soil_type
):

    input_df = prepare_input(
        rainfall,
        irrigation,
        fertilizer,
        temperature,
        water_usage,
        soil_type
    )


    # Scale input
    input_scaled = scaler.transform(
        input_df
    )


    # Prediction
    prediction = model.predict(
        input_scaled
    )


    return prediction[0]


# ============================================================
# PREDICT BUTTON
# ============================================================

if st.button(
    "🌾 Predict Sugarcane Yield",
    use_container_width=True
):

    prediction = predict_sugarcane_yield(

        rainfall,

        irrigation,

        fertilizer,

        temperature,

        water_usage,

        soil_type
    )


    st.success(
        "Prediction completed successfully!"
    )


    # ========================================================
    # RESULT
    # ========================================================

    st.subheader(
        "🌾 Predicted Sugarcane Yield"
    )


    st.metric(
        label="Expected Yield",
        value=f"{prediction:.2f} tons/ha"
    )


    # ========================================================
    # INPUT SUMMARY
    # ========================================================

    st.divider()

    st.subheader(
        "📊 Input Summary"
    )


    result_df = pd.DataFrame({

        "Parameter": [

            "Rainfall",

            "Irrigation",

            "Fertilizer",

            "Temperature",

            "Water Usage",

            "Soil Type"
        ],

        "Value": [

            f"{rainfall:.2f} mm",

            f"{irrigation:.2f} L/ha",

            f"{fertilizer:.2f} kg/ha",

            f"{temperature:.2f} °C",

            f"{water_usage:.2f} L/ha",

            soil_type
        ]
    })


    st.table(
        result_df
    )


# ============================================================
# SIDEBAR
# ============================================================

with st.sidebar:

    st.header("📌 About Project")

    st.write(
        """
        This project predicts sugarcane yield
        using Machine Learning.
        """
    )

    st.write(
        """
        ### Mathematics Used

        • Vectors

        • Matrices

        • Dot Product

        • Matrix Multiplication

        • Derivatives

        • Gradients

        • Gradient Descent

        • Eigenvalues

        • Eigenvectors

        • PCA
        """
    )

    st.divider()

    st.write(
        "Dataset: sugarcane_yield_dataset.csv"
    )

    st.write(
        "Target: Yield_tons_ha"
    )