import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
import json
import torch

# torch device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


# Model names
PYTORCH_MODEL_NAME = 'ahd-cnn-torch.pt'
TENSORFLOW_MODEL_NAME = 'adh-tf.h5'

# Model paths
PYTORCH_AHD_MODEL_PATH = os.path.join(os.getcwd(),
                                      f"models/pytorch/static/{PYTORCH_MODEL_NAME}"
                                      )
TENSORFLOW_AHD_MODEL_PATH = os.path.join(os.getcwd(),
                                      f"models/tensorflow/static/{TENSORFLOW_MODEL_NAME}"
                                      )

# Text vocabularies
with open(os.path.join(os.getcwd(), f"models/pytorch/static/vocab-pt.json"), 'r') as ref:
    PYTORCH_AHD_VOCAB = json.load(ref)
    
with open(os.path.join(os.getcwd(), f"models/tensorflow/static/vocab-tf.json"), 'r') as ref:
    TENSORFLOW_AHD_VOCAB = json.load(ref)
    
# Classes
CLASSES = ["NOT HUMOUR", "HUMOUR"]

# Prediction Type

class PredictionType:
    def __init__(self, label:int, probability:float, class_: str, sent:str) -> None:
        self.label = label
        self.class_ = class_
        self.probability = probability
        self.text = sent
        
    def __str__(self) -> str:
        return "<Prediction Type>"
    
    def to_json(self):
        return {
            'label': self.label,
            'class_': self.class_,
            'probability': self.probability,
            'text': self.text
        }