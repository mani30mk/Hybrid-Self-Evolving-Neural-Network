import joblib
import json
from sentence_transformers import SentenceTransformer

model = joblib.load("../model/model_category_classifier/classifier.pkl")
encoder = SentenceTransformer("../model/model_category_classifier/encoder")

with open("../model/model_category_classifier/label_map.json") as f:
    label_map = json.load(f)

def predict_category(text):
    encoded = encoder.encode([text])
    pred = model.predict(encoded)[0]
    return pred




# from tensorflow.keras.preprocessing.text import Tokenizer
# from tensorflow.keras.preprocessing.text import tokenizer_from_json
# from tensorflow.keras.preprocessing.sequence import pad_sequences
# import numpy as np

# with open("C:/Users/abhin/OneDrive/Desktop/machine_learning/ML_Projects/DeepLearning_Neural_Network/evolving_model_with_algo/dataset/category_tokenizer.json", "r", encoding="utf-8") as f:
#     tokenizer = tokenizer_from_json(f.read())

# CAT_MAP = {
#     0: "Image Classification",
#     1: "Image Segmentation",
#     2: "Image Generation (Generator Only)",
#     3: "Natural Language Processing",
#     4: "Time Series / Audio",
#     5: "Object Detection"
# }

# def predict_category(text, model):
#     seq = pad_sequences(
#         tokenizer.texts_to_sequences([text]),
#         maxlen=40
#     )
#     pred = model.predict(seq)
#     cat_id = np.argmax(pred)
#     return CAT_MAP[cat_id]
