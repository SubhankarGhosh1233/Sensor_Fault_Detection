'''
from sensor.exception import SensorException
from sensor.logger import logging
from sensor.predictor import ModelResolver
import pandas as pd
from sensor.utils import load_object
import os,sys
from datetime import datetime
from sensor.config import TARGET_COLUMN
import numpy as np


PREDICTION_DIR="prediction"



def start_batch_prediction(input_file_path):
    try:
        os.makedirs(PREDICTION_DIR,exist_ok =True)
        logging.info(f"Creating model resolver objact")
        model_resolver = ModelResolver(model_registry="saved_models")
        logging.info(f"Reading file :{input_file_path}")
        df=pd.read_csv(input_file_path)
        df.replace({"na":np.nan},inplace=True)

        # validation are enpty

        logging.info(f"Loading transformer to transform dataset")
        transformer =load_object(file_path=model_resolver.get_latest_transformer_path())

        input_feature_name =[col for col in df.columns if col != TARGET_COLUMN]
        input_arr =transformer.transform(df[input_feature_name])


        logging.info(f"Loading models to make prediction")
        model =load_object(file_path=model_resolver.get_latest_model_path())
        prediction =model.predict(input_arr)

        logging.info(f"target encoder to convert pridivted column into categorical")
        target_encoder =load_object(file_path=model_resolver.get_latest_target_encoder_path())

        cat_prediction =target_encoder.inverse_transform(prediction)

        df["prediction"]=prediction
        df["cat_pred"]=cat_prediction

        prediction_file_name = os.path.basename(input_file_path).replace(".csv",f"{datetime.now().strftime('%m%d%Y__%H%M%S')}.csv")
        prediction_file_path = os.path.join(PREDICTION_DIR,prediction_file_name)
        df.to_csv(prediction_file_path,index=False,header=True)
        logging.info(f"Prediction saved at: {prediction_file_path}")
        return prediction_file_path
    
    except Exception as e:
            raise SensorException(e,sys)
'''
