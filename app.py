from flask import Flask, render_template, request
import pandas as pd
import pickle
from sklearn.ensemble import RandomForestClassifier

app = Flask(__name__)

# Load the trained model
with open("model.pkl", "rb") as f:
    model = pickle.load(f)

@app.route("/")
def index():
    return render_template("upload.html")

@app.route("/predict", methods=["POST"])
def predict():
    input_data = request.form
    try:
        features = [float(input_data[key]) for key in input_data]
        prediction = model.predict([features])[0]
        result = "Demented" if prediction == 1 else "Nondemented"
        return f"<p class='result-text'>Prediction: <strong>{result}</strong></p>"
    except Exception as e:
        return f"<p>Error: {str(e)}</p>"

if __name__ == "__main__":
    app.run(debug=True)
