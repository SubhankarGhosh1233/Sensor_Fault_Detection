from flask import Flask, request, jsonify,render_template,url_for,redirect
from collections import Counter
import pandas as pd
import numpy as np
import os
from sensor.utils import load_object
from sensor.config import TARGET_COLUMN
from sensor.predictor import ModelResolver

app = Flask(__name__)


@app.route('/upload', methods=['GET', 'POST'])
def upload_form():
    if request.method == 'POST':
        try:
            file = request.files['file']
            df = pd.read_csv(file)
            df.replace({"na": np.nan}, inplace=True)

            model_resolver = ModelResolver(model_registry="saved_models")
            transformer = load_object(model_resolver.get_latest_transformer_path())
            model = load_object(model_resolver.get_latest_model_path())
            target_encoder = load_object(model_resolver.get_latest_target_encoder_path())

            input_features = [col for col in df.columns if col != TARGET_COLUMN]
            input_arr = transformer.transform(df[input_features])
            predictions = model.predict(input_arr)
            categories = target_encoder.inverse_transform(predictions)

            df["prediction"] = predictions
            df["category"] = categories

            
            results = df[["prediction", "category"]].to_dict(orient="records")
            category_counts = dict(Counter(df["category"]))
            return render_template("upload.html", results=results, category_counts=category_counts)

        except Exception as e:
            return render_template("upload.html", results=[], error=str(e))

    return render_template("upload.html")

@app.route('/predict', methods=['POST'])
def predict():
    try:
        model_resolver = ModelResolver(model_registry="saved_models")
        transformer = load_object(model_resolver.get_latest_transformer_path())
        model = load_object(model_resolver.get_latest_model_path())
        target_encoder = load_object(model_resolver.get_latest_target_encoder_path())

        file = request.files['file']
        df = pd.read_csv(file)
        df.replace({"na": np.nan}, inplace=True)

        input_features = [col for col in df.columns if col != TARGET_COLUMN]
        input_arr = transformer.transform(df[input_features])
        predictions = model.predict(input_arr)
        categories = target_encoder.inverse_transform(predictions)

        df["prediction"] = predictions
        df["category"] = categories

        result = df[["prediction", "category"]].to_dict(orient="records")

        model_version = model_resolver.get_latest_dir_path().split(os.sep)[-1]
        return jsonify({
            "model_version": model_version,
            "results": result
        })
        
    except Exception as e:
        return jsonify({"error": str(e)}), 500
@app.route('/')
def index():
    return redirect(url_for('upload_form'))

if __name__ == "__main__":
    print("App running at http://127.0.0.1:5000/upload")
    app.run(host="0.0.0.0",port=5000,debug=True)
