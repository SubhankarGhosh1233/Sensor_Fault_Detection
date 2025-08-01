
from sensor.predictor import ModelResolver
from sensor.entity import config_entity,artifact_entity
from sensor.exception import SensorException
from sensor.logger import logging
import sys,os
from sensor.utils import load_object
from sklearn.metrics import f1_score
import pandas as pd
from sensor.config import TARGET_COLUMN


class ModelEvaluation:
    def __init__(self,
        model_eval_config:config_entity.ModelEvaluationConfig,
        data_ingestion_artifact:artifact_entity.DataIngestionArtifact,
        data_transformation_artifact:artifact_entity.DataTransformationArtifact,
        model_trainer_artifact:artifact_entity.ModelTrainerArtifact      
        ):
        try:
            logging.info(f"{'>>'*20}  Model Evaluation {'<<'*20}")
            self.model_eval_config=model_eval_config
            self.data_ingestion_artifact=data_ingestion_artifact
            self.data_transformation_artifact=data_transformation_artifact
            self.model_trainer_artifact=model_trainer_artifact
            self.model_resolver = ModelResolver()
        except Exception as e:
            raise SensorException(e,sys)
        
    def initiate_model_evaluation(self)->artifact_entity.ModelEvaluationArtifact:
        try:
            # if saved model folder has model the we will compare 
            # which model is best trained or the model from saved model folder
            
            logging.info("if saved model folder has model the we will compare" 
            "which model is best trained or the model from saved model folder")
            latest_dir_path = self.model_resolver.get_latest_dir_path()
            if latest_dir_path==None:
                model_eval_artifact = artifact_entity.ModelEvaluationArtifact(is_model_accepted=True,
                improved_accuracy=None)
                logging.info(f"Model evaluation artifact: {model_eval_artifact}")
                return model_eval_artifact

            ## finding location of transformer model and target encoder 
            logging.info("Finding location of transformer model and target encoder")
            transformer_path=self.model_resolver.get_latest_transformer_path()
            model_path = self.model_resolver.get_latest_model_path()
            target_encoder_path =self.model_resolver.get_latest_target_encoder_path()

            logging.info("Previously trained objects of transformer, model and target encoder")
            #1
            ## previously trained (loading) objects
            transformer= load_object(file_path=transformer_path)
            model =load_object(file_path=model_path)
            target_encoder = load_object(file_path=target_encoder_path)

            logging.info("Currently trained model objects")
            #2
            ## currently trained model objects
            current_tratransformer=load_object(file_path=self.data_transformation_artifact.transform_object_path)
            current_model =load_object(file_path=self.model_trainer_artifact.model_path)
            current_terget_encoder =load_object(file_path=self.data_transformation_artifact.target_encoder_path)

            ### comperison
            test_df =pd.read_csv(self.data_ingestion_artifact.test_file_path)
            target_df = test_df[TARGET_COLUMN]
            y_true =target_encoder.transform(target_df)

              # accuracy using previous trained   
            input_feature_name= test_df.drop(TARGET_COLUMN,axis=1).columns.tolist()
            input_arr = transformer.transform(test_df[input_feature_name])  
            y_pred= model.predict(input_arr)
            print(f"Prediction using previous model: {target_encoder.inverse_transform(y_pred[:5])}")
            previous_model_score =f1_score(y_true=y_true,y_pred=y_pred)
            logging.info(f"Accuracy using previous trained model: {previous_model_score}")


              # accuracy using current trained 
            input_feature_name= [col for col in test_df.columns if col != TARGET_COLUMN]
            input_arr = current_tratransformer.transform(test_df[input_feature_name])  
            y_pred= current_model.predict(input_arr)
            y_true =current_terget_encoder.transform(target_df)

            print(f"Prediction using previous model: {current_terget_encoder.inverse_transform(y_pred[:5])}")
            current_model_score =f1_score(y_true=y_true,y_pred=y_pred)
            logging.info(f"Accuracy using current trained: {current_model_score}")

            if current_model_score<=previous_model_score:
                logging.info("Current trained model is not better than previous model")
                raise Exception("Current trained model is not better than previous model")
            
            model_eval_artifact =artifact_entity.ModelEvaluationArtifact(is_model_accepted=True,
            improved_accuracy=current_model_score-previous_model_score)
            logging.info(f"model evel artifact: {model_eval_artifact}")
            return model_eval_artifact


        except Exception as e:
            raise SensorException(e,sys)
