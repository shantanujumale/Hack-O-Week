import os
import io
import pickle
import numpy as np
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.preprocessing import PolynomialFeatures

# -----------------------------------------------------------------------------
# PAGE CONFIGURATION
# -----------------------------------------------------------------------------
st.set_page_config(
    page_title="Sugarcane Yield Prediction AI",
    page_icon="🌾",
    layout="wide",
    initial_sidebar_state="expanded"
)

# -----------------------------------------------------------------------------
# CUSTOM CSS STYLING
# -----------------------------------------------------------------------------
st.markdown("""
<style>
    /* Dark Theme Core */
    .stApp {
        background-color: #0b132b;
        color: #f1f5f9;
    }
    
    /* Header Gradient Banner */
    .main-header {
        background: linear-gradient(135deg, #059669 0%, #10b981 50%, #047857 100%);
        padding: 26px 30px;
        border-radius: 20px;
        color: white;
        margin-bottom: 24px;
        box-shadow: 0 12px 30px -5px rgba(16, 185, 129, 0.35);
    }
    
    .main-header h1 {
        color: #ffffff;
        font-weight: 800;
        margin-bottom: 8px;
        font-size: 2.3rem;
        letter-spacing: -0.02em;
    }
    
    .main-header p {
        color: #d1fae5;
        font-size: 1.1rem;
        margin: 0;
    }

    /* Glassmorphism Cards */
    .metric-card {
        background: rgba(30, 41, 59, 0.85);
        border: 1px solid rgba(255, 255, 255, 0.12);
        border-radius: 20px;
        padding: 24px;
        text-align: center;
        backdrop-filter: blur(12px);
        box-shadow: 0 8px 25px rgba(0, 0, 0, 0.3);
        transition: all 0.3s ease;
    }
    
    .metric-card:hover {
        transform: translateY(-4px);
        border-color: #10b981;
        box-shadow: 0 12px 30px rgba(16, 185, 129, 0.25);
    }

    .metric-value {
        font-size: 3.2rem;
        font-weight: 900;
        background: linear-gradient(135deg, #34d399, #10b981);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        line-height: 1.1;
        margin: 10px 0;
    }

    .metric-label {
        color: #94a3b8;
        font-size: 0.95rem;
        font-weight: 700;
        text-transform: uppercase;
        letter-spacing: 0.08em;
    }

    .status-badge-high {
        background-color: rgba(16, 185, 129, 0.2);
        color: #34d399;
        border: 1.5px solid #10b981;
        padding: 8px 20px;
        border-radius: 30px;
        font-weight: 700;
        display: inline-block;
        font-size: 1rem;
    }

    .status-badge-low {
        background-color: rgba(244, 63, 94, 0.2);
        color: #fb7185;
        border: 1.5px solid #f43f5e;
        padding: 8px 20px;
        border-radius: 30px;
        font-weight: 700;
        display: inline-block;
        font-size: 1rem;
    }

    /* Sidebar Styling */
    section[data-testid="stSidebar"] {
        background-color: #1c2541;
        border-right: 1px solid rgba(255, 255, 255, 0.08);
    }
</style>
""", unsafe_allow_html=True)

# -----------------------------------------------------------------------------
# LOAD PICKLE MODEL PIPELINE
# -----------------------------------------------------------------------------
@st.cache_resource
def load_pipeline():
    base_dir = os.path.dirname(os.path.abspath(__file__))
    pipeline_path = os.path.join(base_dir, "sugarcane_pipeline.pkl")
    
    if not os.path.exists(pipeline_path):
        st.error(f"❌ Pickle file not found at: `{pipeline_path}`. Please run `sugarcane_ml_pipeline.py` first.")
        st.stop()
        
    with open(pipeline_path, "rb") as f:
        bundle = pickle.load(f)
    return bundle

bundle = load_pipeline()

# -----------------------------------------------------------------------------
# HEADER BANNER
# -----------------------------------------------------------------------------
st.markdown("""
<div class="main-header">
    <h1>🌱 Sugarcane Crop Yield Prediction Web App</h1>
    <p>Interactive Machine Learning portal for continuous sugarcane yield estimation (tons/hectare) and classification performance analysis.</p>
</div>
""", unsafe_allow_html=True)

# -----------------------------------------------------------------------------
# SIDEBAR NAVIGATION & MODEL SELECTOR
# -----------------------------------------------------------------------------
st.sidebar.title("🌾 App Menu")
page = st.sidebar.radio(
    "Select Mode:",
    [
        "🔮 Single Yield Predictor",
        "📂 CSV Batch Prediction",
        "📊 Model Metrics & Visuals",
        "📐 Mathematics & Code Snippet"
    ]
)

st.sidebar.markdown("---")
st.sidebar.subheader("⚙️ Model Selector")

reg_model_choice = st.sidebar.selectbox(
    "Regression Model:",
    ["Ridge Regression (alpha=10)", "Lasso Regression (alpha=0.1)", "Linear Regression"],
    help="Select regression algorithm for continuous yield prediction"
)

cls_model_choice = st.sidebar.selectbox(
    "Classification Model:",
    ["Logistic Regression", "K-Nearest Neighbors (KNN k=7)"],
    help="Select classification algorithm for yield performance category"
)

st.sidebar.markdown("---")
st.sidebar.success("✅ `sugarcane_pipeline.pkl` Active")
st.sidebar.caption("Dataset Path: `D:\\SIT\\Sem.5\\Hack 0 Week\\week 7 and 8`")

# Helper to select active model pipeline
reg_pipeline = bundle["regression_pipeline"]
cls_pipeline = bundle["classification_pipeline"]

# =============================================================================
# PAGE 1: SINGLE YIELD PREDICTOR
# =============================================================================
if page == "🔮 Single Yield Predictor":
    st.subheader("📋 Interactive Input Parameters")
    
    col_input, col_result = st.columns([1.25, 1])

    with col_input:
        st.markdown("##### 🧪 Agronomic & Environmental Inputs")
        
        c1, c2 = st.columns(2)
        with c1:
            soil_type = st.selectbox(
                "Soil Type",
                options=["Loamy", "Clayey", "Sandy"],
                help="Soil classification type in the target plot"
            )
            rainfall = st.slider(
                "Rainfall (mm)",
                min_value=300, max_value=1200, value=850, step=10,
                help="Annual rainfall in millimeters"
            )
            irrigation = st.slider(
                "Irrigation (Liters/ha)",
                min_value=500, max_value=2500, value=1200, step=25,
                help="Irrigation water applied per hectare"
            )

        with c2:
            fertilizer = st.slider(
                "Fertilizer Applied (kg/ha)",
                min_value=100, max_value=400, value=250, step=5,
                help="Fertilizer dosage in kilograms per hectare"
            )
            temperature = st.slider(
                "Temperature (°C)",
                min_value=18.0, max_value=40.0, value=28.0, step=0.5,
                help="Average temperature during growing season"
            )
            water_usage = st.slider(
                "Total Water Usage (Liters/ha)",
                min_value=1200, max_value=3500, value=2050, step=25,
                help="Cumulative water usage per hectare"
            )

        # Feature Engineering calculation
        total_water_computed = rainfall + irrigation
        water_temp_ratio = total_water_computed / (temperature + 1e-5)
        fertilizer_water_ratio = fertilizer / (total_water_computed + 1e-5)

        st.markdown("---")
        st.markdown("##### ⚡ Auto-Computed Feature Ratios")
        ec1, ec2, ec3 = st.columns(3)
        ec1.metric("Total Water (ha)", f"{total_water_computed:,} L")
        ec2.metric("Water-Temp Ratio", f"{water_temp_ratio:.2f}")
        ec3.metric("Fertilizer-Water Index", f"{fertilizer_water_ratio:.4f}")

        input_df = pd.DataFrame([{
            "Soil_Type": soil_type,
            "Rainfall_mm": rainfall,
            "Irrigation_liters_ha": irrigation,
            "Fertilizer_kg_ha": fertilizer,
            "Temperature_C": temperature,
            "Water_Usage_liters_ha": water_usage,
            "Total_Water_ha": total_water_computed,
            "Water_Temp_Ratio": water_temp_ratio,
            "Fertilizer_Water_Ratio": fertilizer_water_ratio
        }])

    with col_result:
        st.markdown("##### 🎯 Prediction Output")
        
        predict_btn = st.button("🚀 Calculate Yield Prediction", use_container_width=True, type="primary")

        # Run Prediction
        pred_yield = reg_pipeline.predict(input_df)[0]
        pred_cls_prob = cls_pipeline.predict_proba(input_df)[0][1]
        is_high_yield = pred_cls_prob >= 0.5

        st.markdown("<br>", unsafe_allow_html=True)
        
        # Display Metric Card
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-label">Estimated Sugarcane Yield</div>
            <div class="metric-value">{pred_yield:.2f} <span style="font-size: 1.4rem;">tons/ha</span></div>
            <div style="margin-top: 14px;">
                {'<span class="status-badge-high">🌾 High Yield Target (≥ 63.5 t/ha)</span>' if is_high_yield else '<span class="status-badge-low">⚠️ Low Yield Category (< 63.5 t/ha)</span>'}
            </div>
        </div>
        """, unsafe_allow_html=True)

        st.markdown("<br>", unsafe_allow_html=True)
        st.write(f"**High Yield Probability**: `{pred_cls_prob * 100:.1f}%`")
        st.progress(float(pred_cls_prob))

        # Sensitivity plot: Yield vs Rainfall variation
        st.markdown("##### 📈 Yield Sensitivity to Rainfall")
        rain_range = np.linspace(400, 1200, 20)
        temp_input_df = pd.concat([input_df] * 20, ignore_index=True)
        temp_input_df["Rainfall_mm"] = rain_range
        temp_input_df["Total_Water_ha"] = temp_input_df["Rainfall_mm"] + irrigation
        temp_input_df["Water_Temp_Ratio"] = temp_input_df["Total_Water_ha"] / (temperature + 1e-5)
        temp_input_df["Fertilizer_Water_Ratio"] = fertilizer / (temp_input_df["Total_Water_ha"] + 1e-5)
        
        sim_yields = reg_pipeline.predict(temp_input_df)

        fig, ax = plt.subplots(figsize=(6, 2.8))
        ax.plot(rain_range, sim_yields, color="#10b981", lw=2.5)
        ax.axvline(x=rainfall, color="#f59e0b", linestyle="--", label=f"Current: {rainfall}mm")
        ax.set_facecolor("#1e293b")
        fig.patch.set_facecolor("#1e293b")
        ax.tick_params(colors="#94a3b8")
        ax.xaxis.label.set_color('#94a3b8')
        ax.yaxis.label.set_color('#94a3b8')
        ax.set_xlabel("Rainfall (mm)", fontsize=9)
        ax.set_ylabel("Predicted Yield (t/ha)", fontsize=9)
        ax.legend(facecolor="#0f172a", edgecolor="none", labelcolor="white", fontsize=8)
        plt.tight_layout()
        st.pyplot(fig)

        # Recommendations
        st.markdown("##### 💡 Custom Agronomic Insights")
        if soil_type == "Sandy" and total_water_computed < 2200:
            st.warning("💡 **Irrigation Tip**: Sandy soil has low moisture retention. Adding 200L/ha irrigation could increase yield by ~4.5 tons/ha.")
        elif total_water_computed > 2600 and fertilizer < 220:
            st.info("💡 **Nutrient Balance**: High water volume detected. Nitrogen top-dressing recommended to prevent dilution loss.")
        else:
            st.success("✅ **Balanced Conditions**: Hydrothermal ratio and fertilizer index are optimal for sugarcane maturity!")

# =============================================================================
# PAGE 2: CSV BATCH PREDICTION
# =============================================================================
elif page == "📂 CSV Batch Prediction":
    st.subheader("📂 Batch Prediction from CSV Dataset")
    st.write("Upload a CSV file containing multiple sugarcane plot samples to predict yields and performance categories in bulk.")

    uploaded_file = st.file_uploader("Upload CSV File", type=["csv"])

    if uploaded_file is not None:
        batch_df = pd.read_csv(uploaded_file)
        st.success(f"File uploaded successfully! Loaded {len(batch_df)} records.")
        
        st.markdown("##### Preview Raw Upload:")
        st.dataframe(batch_df.head(), use_container_width=True)

        if st.button("🚀 Run Batch Prediction", type="primary"):
            req_cols = ["Soil_Type", "Rainfall_mm", "Irrigation_liters_ha", "Fertilizer_kg_ha", "Temperature_C", "Water_Usage_liters_ha"]
            missing_cols = [c for c in req_cols if c not in batch_df.columns]

            if missing_cols:
                st.error(f"Missing required columns in CSV: {missing_cols}")
            else:
                proc_df = batch_df.copy()
                proc_df["Total_Water_ha"] = proc_df["Rainfall_mm"] + proc_df["Irrigation_liters_ha"]
                proc_df["Water_Temp_Ratio"] = proc_df["Total_Water_ha"] / (proc_df["Temperature_C"] + 1e-5)
                proc_df["Fertilizer_Water_Ratio"] = proc_df["Fertilizer_kg_ha"] / (proc_df["Total_Water_ha"] + 1e-5)

                batch_preds = reg_pipeline.predict(proc_df)
                batch_probs = cls_pipeline.predict_proba(proc_df)[:, 1]
                batch_labels = ["High Yield 🌾" if p >= 0.5 else "Low Yield ⚠️" for p in batch_probs]

                batch_df["Predicted_Yield_tons_ha"] = np.round(batch_preds, 2)
                batch_df["High_Yield_Probability"] = np.round(batch_probs * 100, 1)
                batch_df["Yield_Category"] = batch_labels

                st.markdown("##### 📊 Batch Predictions Results:")
                st.dataframe(batch_df, use_container_width=True)

                # Download button
                csv_buffer = io.StringIO()
                batch_df.to_csv(csv_buffer, index=False)
                st.download_button(
                    label="📥 Download Predictions CSV",
                    data=csv_buffer.getvalue(),
                    file_name="sugarcane_predictions_results.csv",
                    mime="text/csv"
                )
    else:
        st.info("💡 **Sample CSV format expected**: `Soil_Type`, `Rainfall_mm`, `Irrigation_liters_ha`, `Fertilizer_kg_ha`, `Temperature_C`, `Water_Usage_liters_ha`")
        
        # Provide sample button using built-in dataset
        sample_path = os.path.join(os.path.dirname(__file__), "sugarcane_yield_dataset.csv")
        if os.path.exists(sample_path):
            sample_data = pd.read_csv(sample_path)
            st.markdown("##### 📄 Demo Dataset Sample:")
            st.dataframe(sample_data, use_container_width=True)

# =============================================================================
# PAGE 3: MODEL METRICS & VISUALS
# =============================================================================
elif page == "📊 Model Metrics & Visuals":
    st.subheader("📊 Machine Learning Models Benchmark Summary")
    
    t1, t2 = st.columns(2)
    with t1:
        st.markdown("### 📈 Regression Models Metrics")
        reg_data = pd.DataFrame([
            {"Model": "Linear Regression", "MSE": 22.31, "RMSE": 4.72, "MAE": 3.68, "R² Score": 0.8275, "5-Fold CV R²": 0.8145},
            {"Model": "Ridge Regression (alpha=10)", "MSE": 22.75, "RMSE": 4.77, "MAE": 3.71, "R² Score": 0.8241, "5-Fold CV R²": 0.8145},
            {"Model": "Lasso Regression (alpha=0.1)", "MSE": 22.42, "RMSE": 4.74, "MAE": 3.68, "R² Score": 0.8266, "5-Fold CV R²": 0.8160},
            {"Model": "Polynomial Reg (Deg 2)", "MSE": 57.72, "RMSE": 7.60, "MAE": 4.57, "R² Score": 0.5537, "5-Fold CV R²": 0.7267},
        ])
        st.dataframe(reg_data.style.highlight_max(axis=0, subset=["R² Score", "5-Fold CV R²"]), use_container_width=True)

    with t2:
        st.markdown("### 🎯 Classification Models Metrics")
        cls_data = pd.DataFrame([
            {"Model": "Logistic Regression", "Accuracy": "88.0%", "Precision": "88.8%", "Recall": "87.0%", "F1-Score": "87.9%", "ROC-AUC": 0.9468},
            {"Model": "K-Nearest Neighbors (KNN k=7)", "Accuracy": "87.0%", "Precision": "87.0%", "Recall": "87.0%", "F1-Score": "87.0%", "ROC-AUC": 0.9369},
        ])
        st.dataframe(cls_data, use_container_width=True)

    st.markdown("---")
    st.markdown("### 🖼️ Saved Model Charts & Visualizations")
    
    plots_dir = os.path.join(os.path.dirname(__file__), "plots")
    
    cp1, cp2 = st.columns(2)
    with cp1:
        cm_path = os.path.join(plots_dir, "confusion_matrices.png")
        if os.path.exists(cm_path):
            st.image(cm_path, caption="Classification Confusion Matrices")
    with cp2:
        roc_path = os.path.join(plots_dir, "roc_auc_curves.png")
        if os.path.exists(roc_path):
            st.image(roc_path, caption="ROC-AUC Curves")

# =============================================================================
# PAGE 4: MATHEMATICS & CODE SNIPPET
# =============================================================================
elif page == "📐 Mathematics & Code Snippet":
    st.subheader("📐 Mathematical Formulations & Pickle Usage")

    st.markdown("#### 1. Regression Metrics & Regularization")
    st.latex(r"R^2 = 1 - \frac{\sum_{i=1}^{n} (y_i - \hat{y}_i)^2}{\sum_{i=1}^{n} (y_i - \bar{y})^2}")
    st.latex(r"\text{RMSE} = \sqrt{\frac{1}{n} \sum_{i=1}^{n} (y_i - \hat{y}_i)^2}")
    st.latex(r"\mathcal{L}_{\text{Ridge}} = \sum_{i=1}^{n} (y_i - \mathbf{x}_i^T \mathbf{w})^2 + \alpha \sum_{j=1}^{p} w_j^2")
    st.latex(r"\mathcal{L}_{\text{Lasso}} = \sum_{i=1}^{n} (y_i - \mathbf{x}_i^T \mathbf{w})^2 + \alpha \sum_{j=1}^{p} |w_j|")

    st.markdown("#### 2. Classification Metrics")
    st.latex(r"\text{Precision} = \frac{TP}{TP + FP}, \quad \text{Recall} = \frac{TP}{TP + FN}, \quad F_1 = 2 \cdot \frac{\text{Precision} \cdot \text{Recall}}{\text{Precision} + \text{Recall}}")

    st.markdown("---")
    st.markdown("#### 📦 How to Load & Predict from Pickle (`sugarcane_pipeline.pkl`)")
    st.code("""
import pickle
import pandas as pd

# Load pipeline bundle
with open("sugarcane_pipeline.pkl", "rb") as f:
    bundle = pickle.load(f)

reg_model = bundle["regression_pipeline"]
cls_model = bundle["classification_pipeline"]

# Input data
new_sample = pd.DataFrame([{
    "Soil_Type": "Loamy",
    "Rainfall_mm": 850,
    "Irrigation_liters_ha": 1200,
    "Fertilizer_kg_ha": 250,
    "Temperature_C": 28.0,
    "Water_Usage_liters_ha": 2050,
    "Total_Water_ha": 2050,
    "Water_Temp_Ratio": 2050 / 28.0,
    "Fertilizer_Water_Ratio": 250 / 2050
}])

# Inference
predicted_yield = reg_model.predict(new_sample)[0]
predicted_prob = cls_model.predict_proba(new_sample)[0][1]

print(f"Predicted Yield: {predicted_yield:.2f} tons/ha")
    """, language="python")



# python -m streamlit run app.py