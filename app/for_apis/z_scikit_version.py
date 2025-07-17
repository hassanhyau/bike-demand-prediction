import joblib

# Load the model
model = joblib.load('models/bike_demand_gradient_boosting_model.pkl')

# Check for specific attributes or version indicators
print(type(model))

# Optionally, check for specific attributes known to exist in certain versions
if hasattr(model, '_loss'):
    print("Model has '_loss' attribute (likely trained with a more recent version of scikit-learn).")
else:
    print("Model does not have '_loss' attribute (likely trained with an older version of scikit-learn).")
