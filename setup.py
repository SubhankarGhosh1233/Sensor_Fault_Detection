from setuptools import find_packages,setup
from typing import List
#from sensor.components.model_pusher import ModelPusherConfig, Model_pusher
#from sensor.entity import artifact_entity, config_entity
#from sensor.utils import load_object
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
# Load previous run artifacts (adjust these paths if needed)
transformer_path = "artifact/<timestamp>/data_transformation/transformer/transformer.pkl"
target_encoder_path = "artifact/<timestamp>/data_transformation/target_encoder/target_encoder.pkl"
model_path = "artifact/<timestamp>/model_trainer/model/model.pkl"

# Load artifact objects manually
transformer = load_object(transformer_path)
target_encoder = load_object(target_encoder_path)
model = load_object(model_path)

# Reconstruct artifact classes
data_transformation_artifact = artifact_entity.DataTransformationArtifact(
    transform_object_path=transformer_path,
    transformed_train_path="",  # not required here
    transformed_test_path="",   # not required here
    target_encoder_path=target_encoder_path
)

model_trainer_artifact = artifact_entity.ModelTrainerArtifact(
    model_path=model_path,
    f1_train_score=0.99,  # dummy scores just for completeness
    f1_test_score=0.98
)

# Create pipeline config
training_pipeline_config = config_entity.TrainingPipelineConfig()
model_pusher_config = config_entity.ModelPusherConfig(training_pipeline_config=training_pipeline_config)

# Run model pusher
model_pusher = Model_pusher(
    model_pusher_config=model_pusher_config,
    data_transformation_artifact=data_transformation_artifact,
    model_trainer_artifact=model_trainer_artifact
)

model_pusher_artifact = model_pusher.initiate_model_pusher()
print(f"Model pushed to: {model_pusher_artifact.saved_model_dir}")
'''