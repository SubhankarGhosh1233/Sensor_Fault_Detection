from setuptools import find_packages,setup
from typing import List

import os
REQUIREMENT_FILE_NAME="requirements.txt"
HYPHEN_E_DOT = "-e ."


def get_requirements()->List[str]:
    
    with open(REQUIREMENT_FILE_NAME) as requirement_file:
        requirement_list = requirement_file.readlines()
    requirement_list = [requirement_name.replace("\n", "") for requirement_name in requirement_list]
    
    if HYPHEN_E_DOT in requirement_list:
        requirement_list.remove(HYPHEN_E_DOT)
    return requirement_list


setup(
    name = "sensor",
    version = "0.0.2",
    author = "subhankar",
    author_email = "subhankarghoshds@gmail.com",
    packages = find_packages(),
    install_requires = get_requirements(),

)
'''

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
            return render_template("upload.html", results=results)

        except Exception as e:
            return render_template("upload.html", results=[], error=str(e))

    return render_template("upload.html")
'''