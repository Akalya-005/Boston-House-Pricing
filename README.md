Boston Housing Price Prediction
This project builds and evaluates machine learning models to predict housing prices using the California Housing dataset (as a modern alternative to the deprecated Boston dataset). The aim is to understand which features influence housing prices and make accurate predictions using regression techniques.

📊 Features Used
Median Income

House Age

Average Rooms

Average Bedrooms

Population

Occupancy

Latitude

Longitude

🎯 Objectives
Preprocess the dataset

Train models using Linear Regression and Ridge Regression

Evaluate model performance using MSE and R² score

Export model coefficients and evaluation metrics for analysis

🧠 Algorithms Used
Linear Regression

Ridge Regression (with regularization)

📁 Project Structure
nginx
Copy
Edit
Boston House Pricing/
│
├── model_training.py              # Trains models and exports results
├── model/
│   ├── linear_model_coeffs.csv    # Linear model feature coefficients
│   ├── ridge_model_coeffs.csv     # Ridge model feature coefficients
│   └── model_scores.json          # Performance scores (MSE, R²)
🛠️ Libraries Used
pandas, numpy — data handling

scikit-learn — ML models and preprocessing

matplotlib, seaborn (optional for visualizations)

📈 Sample Results
json
Copy
Edit
{
  "LinearRegression": {
    "MSE": 0.5559,
    "R2": 0.5758
  },
  "RidgeRegression": {
    "MSE": 0.5559,
    "R2": 0.5758
  }
}
🚀 How to Run
Clone the repo

Install dependencies

Run training script:

bash
Copy
Edit
python model_training.py
✅ Future Improvements
Add data visualizations (heatmaps, scatter plots)

Integrate prediction interface for real-time input

Include hyperparameter tuning via GridSearchCV

🙌 Acknowledgment
Dataset: California Housing Dataset (scikit-learn)

