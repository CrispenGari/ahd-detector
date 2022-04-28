import tensorflow as tf
from tensorflow import keras
from models import TENSORFLOW_AHD_MODEL_PATH, TENSORFLOW_AHD_VOCAB, PredictionType
import nltk
from nltk.tokenize import word_tokenize

nltk.download('punkt')

max_words = 100

print(" ✅ LOADING TENSORFLOW AHD MODEL!\n") 
ahd_model = keras.models.load_model(TENSORFLOW_AHD_MODEL_PATH)
print(" ✅ LOADING TENSORFLOW AHD MODEL DONE!\n")

def text_to_sequence(sent):
    words = word_tokenize(sent.lower())
    sequences = []
    for word in words:
        try:
            sequences.append(TENSORFLOW_AHD_VOCAB[word])
        except:
            sequences.append(0)
    return sequences

def predict_humour(sent: str, model):
    classes =["HUMOUR", "NOT HUMOUR"]
    tokens = text_to_sequence(sent)
    padded_tokens = keras.preprocessing.sequence.pad_sequences([tokens],
                                    maxlen=max_words,
                                    padding="post", 
                                    truncating="post"
                                    )
    
    pred = model.predict(padded_tokens)
    pred = tf.squeeze(pred).numpy()
    label = 1 if pred >=0.5 else 0
    probability = float(round(pred, 3)) if pred >= 0.5 else float(round(1 - pred, 3))
    return PredictionType(label=label, 
                          probability=probability, 
                          class_= classes[label],
                          sent=sent.lower()
                          )