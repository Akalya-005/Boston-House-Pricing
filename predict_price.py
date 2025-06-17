import pickle
import pandas as pd

# Load models
with open("model/ridge_model.pkl", "rb") as f:
    model = pickle.load(f)
with open("model/scaler.pkl", "rb") as f:
    scaler = pickle.load(f)

# Input: Example feature set
# Order: ['MedInc', 'HouseAge', 'AveRooms', 'AveBedrms', 'Population', 'AveOccup', 'Latitude', 'Longitude']
input_data = [8.3252, 41.0, 6.9841, 1.0238, 322.0, 2.5556, 37.88, -122.23]

# Scale input and predict
scaled_input = scaler.transform([input_data])
prediction = model.predict(scaled_input)
print(f"Predicted house price (in $100,000s): {round(prediction[0], 2)}")
