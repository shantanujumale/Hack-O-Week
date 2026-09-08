import os
import pickle
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split, cross_val_score, KFold, StratifiedKFold
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder, PolynomialFeatures
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline

# Regression Models
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

# Classification Models
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, roc_curve, confusion_matrix, classification_report
)

def main():
    print("=" * 70)
    print("  SUGARCANE YIELD PREDICTION MACHINE LEARNING PIPELINE")
    print("=" * 70)

    # ---------------------------------------------------------
    # 1. DATA LOADING
    # ---------------------------------------------------------
    base_dir = os.path.dirname(os.path.abspath(__file__))
    large_csv = os.path.join(base_dir, "datasets", "sugarcane_yield_dataset_large.csv")
    small_csv = os.path.join(base_dir, "sugarcane_yield_dataset.csv")

    if os.path.exists(large_csv):
        data_path = large_csv
    elif os.path.exists(small_csv):
        data_path = small_csv
    else:
        raise FileNotFoundError("No sugarcane dataset found!")

    print(f"\n[1] Loading dataset from: {data_path}")
    df = pd.read_csv(data_path)
    print(f"    Dataset Shape: {df.shape[0]} rows, {df.shape[1]} columns")
    print("\nFirst 5 rows:")
    print(df.head())

    # Drop non-predictive ID column if present
    if "Plot_ID" in df.columns:
        df = df.drop(columns=["Plot_ID"])

    # Introduce synthetic missing values if none exist to demonstrate handling missing data
    missing_count_before = df.isnull().sum().sum()
    if missing_count_before == 0:
        np.random.seed(42)
        # Inject ~2% missing values in Rainfall_mm & Temperature_C
        mask_rain = np.random.rand(len(df)) < 0.02
        mask_temp = np.random.rand(len(df)) < 0.02
        df.loc[mask_rain, "Rainfall_mm"] = np.nan
        df.loc[mask_temp, "Temperature_C"] = np.nan
        print(f"\n[2] Injected missing values to demonstrate imputation: {df.isnull().sum().to_dict()}")
    else:
        print(f"\n[2] Existing missing values: {df.isnull().sum().to_dict()}")

    # ---------------------------------------------------------
    # 2. FEATURE ENGINEERING & PREPROCESSING SETUP
    # ---------------------------------------------------------
    print("\n[3] Performing Feature Engineering...")
    # Add domain specific features
    df["Total_Water_ha"] = df["Rainfall_mm"].fillna(df["Rainfall_mm"].median()) + df["Irrigation_liters_ha"].fillna(df["Irrigation_liters_ha"].median())
    df["Water_Temp_Ratio"] = df["Total_Water_ha"] / (df["Temperature_C"].fillna(df["Temperature_C"].median()) + 1e-5)
    df["Fertilizer_Water_Ratio"] = df["Fertilizer_kg_ha"].fillna(df["Fertilizer_kg_ha"].median()) / (df["Total_Water_ha"] + 1e-5)

    # Classification Target: Binary High Yield (1) vs Low Yield (0) based on median yield
    yield_median = df["Yield_tons_ha"].median()
    df["High_Yield_Class"] = (df["Yield_tons_ha"] >= yield_median).astype(int)
    print(f"    Yield Median: {yield_median:.2f} tons/ha")
    print(f"    Classification Class Balance: {df['High_Yield_Class'].value_counts().to_dict()}")

    # Separate Features and Targets
    X = df.drop(columns=["Yield_tons_ha", "High_Yield_Class"])
    y_reg = df["Yield_tons_ha"]
    y_cls = df["High_Yield_Class"]

    # Categorical and Numerical Columns
    num_cols = X.select_dtypes(include=[np.number]).columns.tolist()
    cat_cols = X.select_dtypes(include=['object', 'category']).columns.tolist()

    print(f"    Numerical Features ({len(num_cols)}): {num_cols}")
    print(f"    Categorical Features ({len(cat_cols)}): {cat_cols}")

    # Build Pipeline Transformers
    num_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='median')),
        ('scaler', StandardScaler())
    ])

    cat_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='most_frequent')),
        ('onehot', OneHotEncoder(handle_unknown='ignore', sparse_output=False))
    ])

    preprocessor = ColumnTransformer(transformers=[
        ('num', num_transformer, num_cols),
        ('cat', cat_transformer, cat_cols)
    ])

    # ---------------------------------------------------------
    # 3. REGRESSION MODELING & EVALUATION
    # ---------------------------------------------------------
    print("\n" + "="*50)
    print(" 4. REGRESSION MODELS (Yield Prediction in tons/ha)")
    print("="*50)

    X_train_r, X_test_r, y_train_r, y_test_r = train_test_split(X, y_reg, test_size=0.2, random_state=42)

    # 1) Linear Regression
    lr_pipeline = Pipeline(steps=[('preprocessor', preprocessor), ('regressor', LinearRegression())])
    lr_pipeline.fit(X_train_r, y_train_r)
    y_pred_lr = lr_pipeline.predict(X_test_r)

    # 2) Polynomial Regression (Degree 2)
    poly_preprocessor = Pipeline(steps=[
        ('preprocessor', preprocessor),
        ('poly', PolynomialFeatures(degree=2, include_bias=False))
    ])
    poly_pipeline = Pipeline(steps=[('preprocessor', poly_preprocessor), ('regressor', LinearRegression())])
    poly_pipeline.fit(X_train_r, y_train_r)
    y_pred_poly = poly_pipeline.predict(X_test_r)

    # 3) Ridge Regression
    ridge_pipeline = Pipeline(steps=[('preprocessor', preprocessor), ('regressor', Ridge(alpha=10.0))])
    ridge_pipeline.fit(X_train_r, y_train_r)
    y_pred_ridge = ridge_pipeline.predict(X_test_r)

    # 4) Lasso Regression
    lasso_pipeline = Pipeline(steps=[('preprocessor', preprocessor), ('regressor', Lasso(alpha=0.1))])
    lasso_pipeline.fit(X_train_r, y_train_r)
    y_pred_lasso = lasso_pipeline.predict(X_test_r)

    # Function to calculate regression metrics
    def evaluate_regression(name, y_true, y_pred, model, X_tr, y_tr):
        mse = mean_squared_error(y_true, y_pred)
        rmse = np.sqrt(mse)
        mae = mean_absolute_error(y_true, y_pred)
        r2 = r2_score(y_true, y_pred)
        
        # 5-Fold Cross Validation R2 score
        kf = KFold(n_splits=5, shuffle=True, random_state=42)
        cv_scores = cross_val_score(model, X_tr, y_tr, cv=kf, scoring='r2')
        
        print(f"\n--- {name} ---")
        print(f"  MSE:  {mse:.4f}")
        print(f"  RMSE: {rmse:.4f}")
        print(f"  MAE:  {mae:.4f}")
        print(f"  R2 Score (Test): {r2:.4f}")
        print(f"  R2 Score (5-Fold CV Mean): {cv_scores.mean():.4f} (+/- {cv_scores.std():.4f})")
        return {"Model": name, "MSE": mse, "RMSE": rmse, "MAE": mae, "R2_Test": r2, "R2_CV_Mean": cv_scores.mean()}

    reg_results = []
    reg_results.append(evaluate_regression("Linear Regression", y_test_r, y_pred_lr, lr_pipeline, X_train_r, y_train_r))
    reg_results.append(evaluate_regression("Polynomial Regression (Deg 2)", y_test_r, y_pred_poly, poly_pipeline, X_train_r, y_train_r))
    reg_results.append(evaluate_regression("Ridge Regression (alpha=10)", y_test_r, y_pred_ridge, ridge_pipeline, X_train_r, y_train_r))
    reg_results.append(evaluate_regression("Lasso Regression (alpha=0.1)", y_test_r, y_pred_lasso, lasso_pipeline, X_train_r, y_train_r))

    reg_df = pd.DataFrame(reg_results)

    # ---------------------------------------------------------
    # 4. CLASSIFICATION MODELING & EVALUATION
    # ---------------------------------------------------------
    print("\n" + "="*50)
    print(" 5. CLASSIFICATION MODELS (High Yield vs Low Yield)")
    print("="*50)

    X_train_c, X_test_c, y_train_c, y_test_c = train_test_split(X, y_cls, test_size=0.2, random_state=42, stratify=y_cls)

    # 1) Logistic Regression
    logreg_pipeline = Pipeline(steps=[('preprocessor', preprocessor), ('classifier', LogisticRegression(random_state=42))])
    logreg_pipeline.fit(X_train_c, y_train_c)
    y_pred_logreg = logreg_pipeline.predict(X_test_c)
    y_proba_logreg = logreg_pipeline.predict_proba(X_test_c)[:, 1]

    # 2) K-Nearest Neighbors (KNN)
    knn_pipeline = Pipeline(steps=[('preprocessor', preprocessor), ('classifier', KNeighborsClassifier(n_neighbors=7))])
    knn_pipeline.fit(X_train_c, y_train_c)
    y_pred_knn = knn_pipeline.predict(X_test_c)
    y_proba_knn = knn_pipeline.predict_proba(X_test_c)[:, 1]

    def evaluate_classification(name, y_true, y_pred, y_proba, model, X_tr, y_tr):
        acc = accuracy_score(y_true, y_pred)
        prec = precision_score(y_true, y_pred)
        rec = recall_score(y_true, y_pred)
        f1 = f1_score(y_true, y_pred)
        roc_auc = roc_auc_score(y_true, y_proba)
        
        skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
        cv_scores = cross_val_score(model, X_tr, y_tr, cv=skf, scoring='roc_auc')

        print(f"\n--- {name} ---")
        print(f"  Accuracy:  {acc:.4f}")
        print(f"  Precision: {prec:.4f}")
        print(f"  Recall:    {rec:.4f}")
        print(f"  F1-Score:  {f1:.4f}")
        print(f"  ROC-AUC:   {roc_auc:.4f}")
        print(f"  ROC-AUC (5-Fold CV Mean): {cv_scores.mean():.4f} (+/- {cv_scores.std():.4f})")
        print("  Confusion Matrix:")
        cm = confusion_matrix(y_true, y_pred)
        print(cm)
        return {
            "Model": name, "Accuracy": acc, "Precision": prec, "Recall": rec,
            "F1_Score": f1, "ROC_AUC": roc_auc, "ROC_AUC_CV": cv_scores.mean(),
            "Confusion_Matrix": cm, "y_proba": y_proba
        }

    cls_results = []
    logreg_res = evaluate_classification("Logistic Regression", y_test_c, y_pred_logreg, y_proba_logreg, logreg_pipeline, X_train_c, y_train_c)
    knn_res = evaluate_classification("K-Nearest Neighbors (KNN k=7)", y_test_c, y_pred_knn, y_proba_knn, knn_pipeline, X_train_c, y_train_c)
    cls_results.extend([logreg_res, knn_res])

    cls_df = pd.DataFrame([{k: v for k, v in r.items() if k not in ['Confusion_Matrix', 'y_proba']} for r in cls_results])

    # ---------------------------------------------------------
    # 5. GENERATE & SAVE EVALUATION CHARTS
    # ---------------------------------------------------------
    plots_dir = os.path.join(base_dir, "plots")
    os.makedirs(plots_dir, exist_ok=True)

    # Plot 1: Regression R2 Comparison
    plt.figure(figsize=(8, 5))
    sns.barplot(data=reg_df, x="R2_Test", y="Model", palette="viridis")
    plt.title("Regression Models Comparison (R2 Score)")
    plt.xlim(0, 1.0)
    plt.xlabel("R2 Score (Test Set)")
    plt.tight_layout()
    plt.savefig(os.path.join(plots_dir, "regression_comparison.png"), dpi=300)
    plt.close()

    # Plot 2: Confusion Matrices
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    sns.heatmap(logreg_res["Confusion_Matrix"], annot=True, fmt='d', cmap='Blues', ax=axes[0])
    axes[0].set_title("Logistic Regression Confusion Matrix")
    axes[0].set_xlabel("Predicted Label")
    axes[0].set_ylabel("True Label")

    sns.heatmap(knn_res["Confusion_Matrix"], annot=True, fmt='d', cmap='Greens', ax=axes[1])
    axes[1].set_title("KNN (k=7) Confusion Matrix")
    axes[1].set_xlabel("Predicted Label")
    axes[1].set_ylabel("True Label")
    plt.tight_layout()
    plt.savefig(os.path.join(plots_dir, "confusion_matrices.png"), dpi=300)
    plt.close()

    # Plot 3: ROC Curves
    plt.figure(figsize=(8, 6))
    fpr_log, tpr_log, _ = roc_curve(y_test_c, y_proba_logreg)
    fpr_knn, tpr_knn, _ = roc_curve(y_test_c, y_proba_knn)

    plt.plot(fpr_log, tpr_log, label=f"Logistic Regression (AUC = {logreg_res['ROC_AUC']:.3f})", lw=2)
    plt.plot(fpr_knn, tpr_knn, label=f"KNN k=7 (AUC = {knn_res['ROC_AUC']:.3f})", lw=2)
    plt.plot([0, 1], [0, 1], 'k--', label='Random Chance')
    plt.xlabel("False Positive Rate (1 - Specificity)")
    plt.ylabel("True Positive Rate (Sensitivity / Recall)")
    plt.title("ROC-AUC Curve - Classification Models")
    plt.legend(loc="lower right")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(plots_dir, "roc_auc_curves.png"), dpi=300)
    plt.close()

    print(f"\n[6] Visual evaluation plots saved to: {plots_dir}")

    # ---------------------------------------------------------
    # 6. PICKLE EXPORT
    # ---------------------------------------------------------
    print("\n" + "="*50)
    print(" 7. EXPORTING TRAINED MODELS TO PICKLE (.pkl)")
    print("="*50)

    # Select best regression model (Ridge) & full pipeline
    best_reg_pipeline = ridge_pipeline
    best_cls_pipeline = logreg_pipeline

    # Save individual components & pipelines
    pickle_reg_path = os.path.join(base_dir, "sugarcane_model.pkl")
    pickle_cls_path = os.path.join(base_dir, "sugarcane_classifier.pkl")
    pickle_full_pipeline_path = os.path.join(base_dir, "sugarcane_pipeline.pkl")

    with open(pickle_reg_path, 'wb') as f:
        pickle.dump(best_reg_pipeline, f)

    with open(pickle_cls_path, 'wb') as f:
        pickle.dump(best_cls_pipeline, f)

    pipeline_artifact = {
        "regression_pipeline": best_reg_pipeline,
        "classification_pipeline": best_cls_pipeline,
        "yield_median_threshold": yield_median,
        "feature_names": list(X.columns),
        "numerical_features": num_cols,
        "categorical_features": cat_cols
    }

    with open(pickle_full_pipeline_path, 'wb') as f:
        pickle.dump(pipeline_artifact, f)

    print(f"  Saved Regression Model Pipeline -> {pickle_reg_path}")
    print(f"  Saved Classification Model Pipeline -> {pickle_cls_path}")
    print(f"  Saved Unified Pipeline Bundle -> {pickle_full_pipeline_path}")

    # ---------------------------------------------------------
    # 7. INFERENCE TEST ON NEW UNSEEN INPUT
    # ---------------------------------------------------------
    print("\n" + "="*50)
    print(" 8. TESTING PREDICTION WITH PICKLE FILE ON NEW INPUT")
    print("="*50)

    with open(pickle_full_pipeline_path, 'rb') as f:
        loaded_bundle = pickle.load(f)

    sample_raw_input = pd.DataFrame([{
        "Soil_Type": "Loamy",
        "Rainfall_mm": 850,
        "Irrigation_liters_ha": 1200,
        "Fertilizer_kg_ha": 250,
        "Temperature_C": 28.0,
        "Water_Usage_liters_ha": 2050,
        "Total_Water_ha": 850 + 1200,
        "Water_Temp_Ratio": (850 + 1200) / 28.0,
        "Fertilizer_Water_Ratio": 250 / (850 + 1200)
    }])

    pred_yield = loaded_bundle["regression_pipeline"].predict(sample_raw_input)[0]
    pred_cls_prob = loaded_bundle["classification_pipeline"].predict_proba(sample_raw_input)[0][1]
    pred_cls_label = "High Yield" if pred_cls_prob >= 0.5 else "Low Yield"

    print("\nSample Custom Input:")
    for k, v in sample_raw_input.iloc[0].items():
        print(f"  {k}: {v}")

    print(f"\nPredictions loaded from {pickle_full_pipeline_path}:")
    print(f"  Predicted Sugarcane Yield: {pred_yield:.2f} tons/ha")
    print(f"  Predicted Yield Class:     {pred_cls_label} (Probability of High Yield: {pred_cls_prob*100:.1f}%)")
    print("\nPipeline execution successfully finished!")

if __name__ == "__main__":
    main()
