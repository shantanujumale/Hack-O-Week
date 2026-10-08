import sys
import nbformat as nbf

def add_ensemble_section_to_existing_notebook():
    notebook_path = "Sugarcane_Yield_Prediction_ML.ipynb"
    nb = nbf.read(notebook_path, as_version=4)
    
    # Remove any existing Section 7 or added cell
    nb.cells = [c for c in nb.cells if "7. 🚀 Advanced Extension: Ensemble Methods" not in c.source and "QUICK ENSEMBLE BENCHMARK" not in c.source]

    md_section = r"""---
## 7. 🚀 Advanced Extension: Ensemble Methods, Bias-Variance Tradeoff & Regularization

### 📌 Task Overview for Students:
1. **Ensemble Methods**:
   - **Bagging**: Parallel bootstrap aggregation (Bagging Regressor, Random Forest) that drastically reduces **Variance** without increasing bias.
   - **Boosting**: Sequential residual learning focusing on **XGBoost** and **LightGBM** that iteratively reduces **Bias** while regularization controls variance.
2. **Bias–Variance Trade-off**:
   - $\text{Expected Generalization Error} = \text{Bias}^2 + \text{Variance} + \sigma^2$.
   - **Underfitting (High Bias)**: The model makes overly simplistic assumptions (e.g. linear model on non-linear responses). Both train and test errors are high.
   - **Overfitting (High Variance)**: The model memorizes training noise (e.g. unpruned deep trees). Training error is low, but test error is high with a large generalization gap.
3. **Regularization ($L_1$ and $L_2$)**:
   - $L_2$ Ridge penalty ($\lambda \|w\|_2^2$) shrinks weights smoothly, mitigating multicollinearity.
   - $L_1$ Lasso penalty ($\lambda \|w\|_1$) prunes uninformative coefficients strictly to zero (feature selection).
   - XGBoost & LightGBM leaf weight regularization (`reg_alpha` for $L_1$, `reg_lambda` for $L_2$) prevents leaf over-specialization and shrinks extreme predictions.

> 📘 **Dedicated Deep-Dive Notebook**:
> For the complete study including theoretical proofs, interactive learning curves, validation curves across depths, regularization path visualizers, and production inference advisory, open:
> 👉 **`Sugarcane_Yield_Prediction_Ensemble_Methods.ipynb`**
"""

    code_section = r"""# -----------------------------------------------------------------------------
# QUICK ENSEMBLE BENCHMARK: BAGGING, RANDOM FOREST, XGBOOST & LIGHTGBM
# -----------------------------------------------------------------------------
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import BaggingRegressor, RandomForestRegressor
import xgboost as xgb
import lightgbm as lgb

ensemble_models = {
    "Baseline Linear (OLS)": LinearRegression(),
    "Baseline Ridge (L2)": Ridge(alpha=10.0, random_state=42),
    "Bagging Regressor": BaggingRegressor(estimator=DecisionTreeRegressor(max_depth=8), n_estimators=100, random_state=42),
    "Random Forest (Bagging)": RandomForestRegressor(n_estimators=100, max_depth=8, random_state=42),
    "XGBoost Regressor (Boosting)": xgb.XGBRegressor(n_estimators=100, learning_rate=0.08, max_depth=4, reg_alpha=0.1, reg_lambda=1.5, random_state=42),
    "LightGBM Regressor (Boosting)": lgb.LGBMRegressor(n_estimators=100, learning_rate=0.08, max_depth=4, num_leaves=15, reg_alpha=0.1, reg_lambda=1.5, random_state=42, verbose=-1)
}

ens_results = []
for name, model in ensemble_models.items():
    pipe = Pipeline([
        ('preprocessor', preprocessor),
        ('model', model)
    ])
    pipe.fit(X_train_r, y_train_r)
    y_tr_pred = pipe.predict(X_train_r)
    y_te_pred = pipe.predict(X_test_r)
    
    tr_r2 = r2_score(y_train_r, y_tr_pred)
    te_r2 = r2_score(y_test_r, y_te_pred)
    rmse = np.sqrt(mean_squared_error(y_test_r, y_te_pred))
    gap = tr_r2 - te_r2
    
    ens_results.append({
        "Model": name,
        "Train R2": tr_r2,
        "Test R2": te_r2,
        "Overfitting Gap (Train - Test)": gap,
        "Test RMSE (tons/ha)": rmse
    })

ens_df = pd.DataFrame(ens_results).sort_values(by="Test R2", ascending=False).reset_index(drop=True)
print("=" * 80)
print("🌾 ENSEMBLE METHODS COMPARATIVE BENCHMARK (SUGARCANE YIELD REGRESSION)")
print("=" * 80)
print(ens_df.to_string(index=False))

# Visual Comparison of Test R2 & Overfitting Gap
fig, ax = plt.subplots(figsize=(10, 4.5))
bars = ax.barh(ens_df["Model"], ens_df["Test R2"], color='#10b981', edgecolor='black', alpha=0.85)
ax.set_xlabel('Test $R^2$ Score (Higher is Better)', fontsize=11)
ax.set_title('Test $R^2$ Score across Linear & Ensemble Models', fontweight='bold', fontsize=12)
ax.set_xlim(0.65, 0.90)
for bar in bars:
    w = bar.get_width()
    ax.text(w + 0.005, bar.get_y() + bar.get_height()/2.0, f'{w:.4f}', va='center', fontweight='bold', fontsize=9.5)
plt.tight_layout()
plt.show()
"""

    nb.cells.append(nbf.v4.new_markdown_cell(md_section))
    nb.cells.append(nbf.v4.new_code_cell(code_section))
    
    with open(notebook_path, "w", encoding="utf-8") as f:
        nbf.write(nb, f)
        
    print(f"Successfully updated Section 7 in '{notebook_path}'!")

if __name__ == "__main__":
    add_ensemble_section_to_existing_notebook()
