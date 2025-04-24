import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

# Load preprocessed data
df = pd.read_csv("data/processed_data.csv")

# Separate features and labels
X = df.drop(columns=["Group"])
y = df["Group"]

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train model
model = RandomForestClassifier()
model.fit(X_train, y_train)

# Get predicted probabilities for the test set
y_pred_proba = model.predict_proba(X_test)  # Get probabilities for both classes (0 = NonDemented, 1 = Demented)
print("Predicted Probabilities (for each class):")
print(y_pred_proba)  # This prints an array of probabilities for both classes (NonDemented, Demented)

# You can also print the probabilities for a specific instance (if you want to check a particular test case)
# For example, printing the probabilities for the first instance in the test set
print("Prediction Probabilities for the first test instance:")
print(f"NonDemented: {y_pred_proba[0][0]:.2f}, Demented: {y_pred_proba[0][1]:.2f}")

# Evaluate accuracy
y_pred = model.predict(X_test)
acc = accuracy_score(y_test, y_pred)
print(f"✅ Model trained. Accuracy: {acc:.2f}")
