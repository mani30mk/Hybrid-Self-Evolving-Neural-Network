import json
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, Model
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.preprocessing.text import Tokenizer

CAT_MAP = {
    "Image Classification": 0,
    "Image Segmentation": 1,
    "Image Generation (Generator Only)": 2,
    "Natural Language Processing": 3,
    "Time Series / Audio": 4,
    "Object Detection": 5
}

NUM_CATEGORIES = len(CAT_MAP)

LAYER_VOCAB = [
    "<PAD>", "<START>", "<END>", "Input", "Dense", "Conv1D", "Conv2D", "Conv3D",
    "MaxPooling1D", "MaxPooling2D", "AveragePooling2D", "BatchNormalization",
    "LayerNormalization", "Dropout", "Flatten", "Embedding", "SimpleRNN",
    "LSTM", "GRU", "Bidirectional", "MultiHeadAttention", "GlobalAveragePooling2D",
    "GlobalMaxPooling2D", "UpSampling2D", "Conv2DTranspose", "GlobalAveragePooling1D",
    "Reshape", "LeakyReLU", "Softmax", "Sigmoid", "Concatenate", "Add",
    "FeatureMap", "BBoxRegressionHead", "ClassPredictionHead", "FeaturePyramid"
]

layer_to_id = {name: i for i, name in enumerate(LAYER_VOCAB)}
id_to_layer = {i: name for name, i in layer_to_id.items()}

def load_and_preprocess(filepath):
    texts, cats_raw, targets = [], [], []

    with open(filepath, "r") as f:
        for line in f:
            data = json.loads(line)
            texts.append(data["text"])
            cats_raw.append(data["category"])
            # Convert layer names to IDs, skipping any typos
            seq = [layer_to_id["<START>"]] + [layer_to_id[l] for l in data["layer_order"] if l in layer_to_id] + [layer_to_id["<END>"]]
            targets.append(seq)

    # Automatically create CAT_MAP to avoid KeyErrors
    unique_cats = sorted(list(set(cats_raw)))
    cat_map = {name: i for i, name in enumerate(unique_cats)}
    print(f"✅ Detected Categories: {cat_map}")

    X_cat_ids = np.array([cat_map[c] for c in cats_raw]).reshape(-1, 1)

    tokenizer = Tokenizer(num_words=10000, oov_token="<OOV>")
    tokenizer.fit_on_texts(texts)

    X_txt = pad_sequences(tokenizer.texts_to_sequences(texts), maxlen=40, padding='post')
    Y = pad_sequences(targets, maxlen=25, padding='post')

    return X_txt, X_cat_ids, Y, tokenizer, len(unique_cats), cat_map

def get_skeleton_from_prompt(model, user_prompt, category_name):
    X_txt, X_cat_ids, Y, tokenizer, num_classes, final_cat_map = load_and_preprocess("../dataset/universal_dl_data_bulletproof.jsonl")

    if category_name not in final_cat_map:
        return "Invalid Category"

    t_seq = pad_sequences(tokenizer.texts_to_sequences([user_prompt]), maxlen=40, padding='post')
    c_id = np.array([final_cat_map[category_name]]).reshape(-1, 1)

    # Initialize with <START>
    d_seq = np.zeros((1, 24))
    d_seq[0, 0] = layer_to_id["<START>"]

    prediction = []
    for i in range(1, 24):
        output = model.predict([t_seq, c_id, d_seq], verbose=0)
        idx = np.argmax(output[0, i-1, :])
        layer = id_to_layer[idx]

        if layer == "<END>": break
        if layer not in ["<PAD>", "<START>"]:
            prediction.append(layer)

        d_seq[0, i] = idx

    return prediction
