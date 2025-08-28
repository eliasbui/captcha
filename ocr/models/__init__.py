from .crnn.model import get_model_checkpoint as get_crnn
from .crnn.traning import get_training as get_crnn_training

def get_model(config: dict):
    if config.model_name == "CRNN_Pytorch":
        return get_crnn()
    if config.model_name == "CRNN_Onnx":
        return None
    return None

def get_training(config: dict):
    if config.model_name == "CRNN_Pytorch":
        return get_crnn_training
    return None