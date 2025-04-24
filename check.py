# check.py
import pickle
import numpy as np

# Load model
with open("model.pkl", "rb") as f:
    model = pickle.load(f)

# Input: [MR Delay, M/F, Age, EDUC, SES, MMSE, CDR, eTIV, nWBV, ASF]
test_input = [[10, 1, 85, 12, 1, 20, 1.0, 1400, 0.65, 1.1]]

# Predict
prediction = model.predict(test_input)[0]
prediction_proba = model.predict_proba(test_input)[0]

# Get class labels in correct order
class_labels = model.classes_  # Should be [0, 1] => [Nondemented, Demented]

# Print results
print(f"Predicted Class: {'Demented' if prediction == 1 else 'Nondemented'}")
print(f"Probability - Nondemented: {prediction_proba[class_labels.tolist().index(0)]:.2f}, "
      f"Demented: {prediction_proba[class_labels.tolist().index(1)]:.2f}")


