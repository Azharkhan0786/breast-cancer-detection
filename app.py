from flask import Flask, request, jsonify, render_template
import numpy as np
import pandas as pd
import sklearn.datasets
from sklearn.linear_model import LogisticRegression

app = Flask(__name__)

# Load dataset
data = sklearn.datasets.load_breast_cancer()
df = pd.DataFrame(data.data, columns=data.feature_names)
df['label'] = data.target

X = df.drop(columns='label')
y = df['label']

model = LogisticRegression(max_iter=10000)
model.fit(X, y)

@app.route('/')
def home():
    return render_template('index.html', features=list(data.feature_names))

@app.route('/predict', methods=['POST'])
def predict():
    try:
        input_data = request.json

        # Only use the 5 features present in HTML
        feature_subset = ["mean radius", "mean texture", "mean perimeter", "mean area", "mean smoothness"]

        values = [float(input_data.get(f, 0)) for f in feature_subset]

        # Create a full feature vector of zeros and replace the first 5 with user inputs
        full_input = [0] * len(data.feature_names)
        for i in range(len(feature_subset)):
            full_input[i] = values[i]

        prediction = model.predict([full_input])[0]
        proba = model.predict_proba([full_input])[0]

        result = "Benign" if prediction == 1 else "Malignant"
        return jsonify({
            "result": result,
            "malignant": round(float(proba[0]), 4),
            "benign": round(float(proba[1]), 4)
        })
    except Exception as e:
        return jsonify({"error": str(e)})



if __name__ == '__main__':
    app.run(debug=True)
