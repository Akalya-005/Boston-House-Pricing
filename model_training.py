import pandas as pd
import numpy as np
from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, r2_score
import json

# Load dataset
data = fetch_california_housing()
X = pd.DataFrame(data.data, columns=data.feature_names)
y = pd.Series(data.target, name="Price")

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Feature scaling
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Linear Regression
linear_model = LinearRegression()
linear_model.fit(X_train_scaled, y_train)
y_pred_lr = linear_model.predict(X_test_scaled)

# Ridge Regression
ridge_model = Ridge(alpha=1.0)
ridge_model.fit(X_train_scaled, y_train)
y_pred_ridge = ridge_model.predict(X_test_scaled)

# Print scores
print(f"Linear Regression - MSE: {mean_squared_error(y_test, y_pred_lr):.4f}, R2 Score: {r2_score(y_test, y_pred_lr):.4f}")
print(f"Ridge Regression - MSE: {mean_squared_error(y_test, y_pred_ridge):.4f}, R2 Score: {r2_score(y_test, y_pred_ridge):.4f}")

# Save Coefficients to CSV
lr_df = pd.DataFrame({
    "Feature": X.columns,
    "Coefficient": linear_model.coef_
})
lr_df.to_csv("model/linear_model_coeffs.csv", index=False)

ridge_df = pd.DataFrame({
    "Feature": X.columns,
    "Coefficient": ridge_model.coef_
})
ridge_df.to_csv("model/ridge_model_coeffs.csv", index=False)

# Save Evaluation Scores to JSON
scores = {
    "LinearRegression": {
        "MSE": round(mean_squared_error(y_test, y_pred_lr), 4),
        "R2": round(r2_score(y_test, y_pred_lr), 4)
    },
    "RidgeRegression": {
        "MSE": round(mean_squared_error(y_test, y_pred_ridge), 4),
        "R2": round(r2_score(y_test, y_pred_ridge), 4)
    }
}
with open("model/model_scores.json", "w") as f:
    json.dump(scores, f, indent=4)
