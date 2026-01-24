from flask import Flask, request, jsonify, send_file
from flask_cors import CORS
import tensorflow as tf
import numpy as np
import pandas as pd
import zipfile
import os
import uuid
import threading
from math import prod
from sklearn.preprocessing import StandardScaler

EVOLUTION_STATUS = {}

import base64
import cv2
from io import BytesIO
from PIL import Image

from tensorflow.keras.preprocessing.text import Tokenizer, tokenizer_from_json
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.preprocessing.image import load_img, img_to_array

# =====================================================
# INTERNAL IMPORTS
# =====================================================
from self_evolving_algo import full_pipeline
from inference_utils import get_skeleton_from_prompt
from dataset_validator import validate_dataset
from category_inference import predict_category

# =====================================================
# FLASK SETUP
# =====================================================
app = Flask(__name__)
CORS(app)

BASE_UPLOAD_DIR = "uploads"
os.makedirs(BASE_UPLOAD_DIR, exist_ok=True)

# =====================================================
# LOAD MODELS
# =====================================================
try:
    # category_model = tf.keras.models.load_model(
    #     "../model/category_predictor.keras", compile=False
    # )

    architecture_model = tf.keras.models.load_model(
        "../model/arch_predict_cat2.keras", compile=False
    )

    with open("../dataset/category_tokenizer.json", "r", encoding="utf-8") as f:
        tokenizer_arch = tokenizer_from_json(f.read())

    print("✅ System Ready: Models and tokenizers loaded.")

except Exception as e:
    print(f"❌ Model loading failed: {e}")

def validate_shapes(category, X, y, input_shape, num_classes):
    if category == "Image Segmentation":
        if np.max(y) >= num_classes:
            return False, "Segmentation mask contains invalid class values."

    if category.startswith("Image") and category != "Image Segmentation":
        if X.shape[1:] != input_shape:
            return False, "Image shape does not match input_shape."

    if category == "Natural Language Processing":
        if len(input_shape) != 1:
            return False, "NLP input_shape must be (sequence_length,)."

    if category == "Time Series / Audio":
        if len(input_shape) != 2:
            return False, "Time Series input_shape must be (timesteps, features)."

    # if y is not None and num_classes:
    #     if len(np.unique(y)) > num_classes:
    #         return False, "num_classes is less than unique labels."

    return True, None

###### Model Inference ######

def numpy_to_base64(img_array):
    img_array = np.clip(img_array * 255, 0, 255).astype("uint8")

    if img_array.ndim == 2:
        img_array = cv2.cvtColor(img_array, cv2.COLOR_GRAY2RGB)

    img = Image.fromarray(img_array)
    buffer = BytesIO()
    img.save(buffer, format="PNG")
    return base64.b64encode(buffer.getvalue()).decode("utf-8")

def load_tokenizer(user_dir):
    tokenizer_path = os.path.join(user_dir, "tokenizer.json")
    if os.path.exists(tokenizer_path):
        with open(tokenizer_path, "r", encoding="utf-8") as f:
            return tokenizer_from_json(f.read())
    return None

def run_inference(
    category,
    model,
    X_val,
    y_val,
    input_shape,
    user_dir
):
    results = {}

    # ====================================================
    # NLP
    # ====================================================
    if category == "Natural Language Processing":

        tokenizer = load_tokenizer(user_dir)
        if tokenizer is None:
            raise ValueError("Tokenizer not found.")

        # X_val_seq = pad_sequences(
        #     tokenizer.texts_to_sequences(X_val),
        #     maxlen=input_shape[0]
        # )

        # preds = model.predict(X_val_seq)
        preds = model.predict(X_val)
        y_pred = np.argmax(preds, axis=-1)

        decoded_X_val = tokenizer.sequences_to_texts(X_val)
        decoded_X = []
        for sen in decoded_X_val:
            sen = sen.split()
            sen = " ".join([word for word in sen if word != "<OOV>"])
            decoded_X.append(sen)

        print(decoded_X)

        results["X_val"] = decoded_X[:50]
        results["y_val"] = y_val[:50].tolist()
        results["y_pred"] = y_pred[:50].tolist()

        return results

    # ====================================================
    # IMAGE CLASSIFICATION
    # ====================================================
    if category == "Image Classification":

        preds = model.predict(X_val)
        y_pred = np.argmax(preds, axis=1)
        print(y_pred)

        results["X_val"] = [numpy_to_base64(x) for x in X_val[:20]]
        results["y_val"] = y_val[:20].tolist()
        results["y_pred"] = y_pred[:20].tolist()

        return results

    # ====================================================
    # IMAGE SEGMENTATION
    # ====================================================
    if category == "Image Segmentation":

        preds = model.predict(X_val)
        masks = np.argmax(preds, axis=-1)

        results["X_val"] = [numpy_to_base64(x) for x in X_val[:10]]
        results["y_val"] = [numpy_to_base64(y.squeeze()) for y in y_val[:10]]
        results["y_pred"] = [numpy_to_base64(m) for m in masks[:10]]

        return results

    # ====================================================
    # IMAGE GENERATION
    # ====================================================
    # ====================================================
    # IMAGE GENERATION
    # ====================================================
    if category == "Image Generation (Generator Only)":

        preds = model.predict(X_val)

        # Check sub-mode
        if X_val.ndim == 2:
            # Text-to-Image: Decode X_val
            tokenizer = load_tokenizer(user_dir)
            if tokenizer:
                decoded_X = tokenizer.sequences_to_texts(X_val[:10])
                # Remove <OOV> or padding artifacts if needed
                results["X_val"] = [t.replace("<OOV>", "").strip() for t in decoded_X]
            else:
                results["X_val"] = ["Tokenizer not found" for _ in range(len(X_val[:10]))]
        else:
            # Image-to-Image: Base64 X_val
            results["X_val"] = [numpy_to_base64(x) for x in X_val[:10]]

        results["y_val"] = [numpy_to_base64(y) for y in y_val[:10]]
        results["y_pred"] = [numpy_to_base64(p) for p in preds[:10]]

        return results

    # ====================================================
    # TIME SERIES
    # ====================================================
    if category == "Time Series / Audio":
        preds = model.predict(X_val)
        if preds.shape[-1] == 1:
            y_pred = preds.flatten().tolist()
        else:
            y_pred = np.argmax(preds, axis=-1).tolist()

        results["X_val"] = X_val[:50].tolist()
        results["y_val"] = y_val[:50].tolist()
        results["y_pred"] = y_pred[:50].tolist()

        return results

    return {}


def time_series_preprocess(path):
    df = pd.read_csv(path).dropna()

    df = df.sample(frac=1, random_state=42).reset_index(drop=True)

    split_idx = int(0.8 * len(df))
    train_df = df.iloc[:split_idx]
    val_df = df.iloc[split_idx:]

    X_raw_train = train_df.drop(columns=["y"]).values
    X_raw_val = val_df.drop(columns=["y"]).values

    y_raw_train = train_df["y"].values
    y_raw_val = val_df["y"].values

    unique_y = np.unique(y_raw_train)
    is_regression = len(unique_y) > 10

    X_scaler = StandardScaler()
    X_train = X_scaler.fit_transform(X_raw_train)
    X_val = X_scaler.transform(X_raw_val)

    if is_regression:
        y_scaler = StandardScaler()
        y_train = y_scaler.fit_transform(y_raw_train.reshape(-1, 1)).flatten()
        y_val = y_scaler.transform(y_raw_val.reshape(-1, 1)).flatten()
        num_classes = 1
        loss_fn = 'mse'
    else:
        y_train = y_raw_train.astype(int)
        y_val = y_raw_val.astype(int)
        num_classes = len(unique_y)
        loss_fn = 'sparse_categorical_crossentropy'

    X_train = X_train.reshape(X_train.shape[0], X_train.shape[1], 1).astype("float32")
    X_val = X_val.reshape(X_val.shape[0], X_val.shape[1], 1).astype("float32")

    return X_train, X_val, y_train, y_val, num_classes, loss_fn


# =====================================================
# CATEGORY PREDICTION API
# =====================================================
@app.route("/api/predict-category", methods=["POST"])
def predict_category_route():
    prompt = request.json.get("prompt", "")
    if not prompt:
        return jsonify({"error": "Prompt is required"}), 400

    # category = predict_category(prompt, category_model)
    category = predict_category(prompt)
    return jsonify({"category": category})

# =====================================================
# ARCHITECTURE + TRAINING API
# =====================================================
@app.route("/api/predict-architecture", methods=["POST"])
def predict_architecture_route():

    request_id = str(uuid.uuid4())
    user_dir = os.path.join(BASE_UPLOAD_DIR, request_id)
    os.makedirs(user_dir, exist_ok=True)

    try:
        # -------------------------------------------------
        # METADATA
        # -------------------------------------------------
        prompt = request.form.get("prompt")
        category = request.form.get("category")
        shape_str = request.form.get("input_shape")
        num_classes = request.form.get("num_classes")

        if not all([prompt, category, shape_str]):
            return jsonify({
                "error": "prompt, category, and input_shape are required."
            }), 400

        try:
            input_shape = tuple(map(int, shape_str.split(",")))
        except:
            return jsonify({
                "error": "Invalid input_shape format. Example: 28,28,1"
            }), 400

        num_classes = int(num_classes) if num_classes else None

        # -------------------------------------------------
        # DATASET FILE
        # -------------------------------------------------
        file = request.files.get("dataset")
        if not file or file.filename == "":
            return jsonify({"error": "Dataset file is required."}), 400

        dataset_path = os.path.join(user_dir, file.filename)
        file.save(dataset_path)

        # -------------------------------------------------
        # DATASET VALIDATION (STRICT)
        # -------------------------------------------------
        is_valid, error = validate_dataset(category, dataset_path)
        if not is_valid:
            return jsonify({"error": error}), 400

        # -------------------------------------------------
        # LOAD DATA
        # -------------------------------------------------
        X_final, y_final = None, None
        ext = os.path.splitext(dataset_path)[1].lower()

        # ---------- CSV (NLP / Time Series) ----------
        if category == "Time Series / Audio":
            pass

        elif ext == ".csv":
            df = pd.read_csv(dataset_path).dropna()

            df = df.sample(frac=1, random_state=42).reset_index(drop=True)
            if category == "Natural Language Processing":
                X_raw = df["X"].astype(str).values
                y_final = df["y"].values if "y" in df.columns else None

                tokenizer = Tokenizer(num_words=10000, oov_token="<OOV>")
                tokenizer.fit_on_texts(X_raw)

                with open(os.path.join(user_dir, "tokenizer.json"), "w", encoding="utf-8") as f:
                    f.write(tokenizer.to_json())

                X_final = pad_sequences(
                    tokenizer.texts_to_sequences(X_raw),
                    maxlen=input_shape[0]
                )

            # else:  # Time Series
            #     y_final = df["y"].values
            #     X_raw = df.drop(columns=["y"]).values
            #     X_final = X_raw.reshape(-1, *input_shape)

            #     unique_y = np.unique(y_final)
            #     is_regression = len(unique_y) > 10

            #     X_scaler = StandardScaler()
            #     X_final = X_scaler.fit_transform(X_raw)

        # ---------- ZIP (Image tasks) ----------
        elif ext == ".zip":

            with zipfile.ZipFile(dataset_path, "r") as z:
                z.extractall(user_dir)

            # IMAGE CLASSIFICATION
            # if category == "Image Classification":
            #     images, labels = [], []
            #     classes = sorted(
            #         d for d in os.listdir(user_dir)
            #         if os.path.isdir(os.path.join(user_dir, d)) and d.isdigit()
            #     )

            #     for cls in classes:
            #         for img_name in os.listdir(os.path.join(user_dir, cls)):
            #             img_path = os.path.join(user_dir, cls, img_name)
            #             try:
            #                 img = load_img(img_path, target_size=input_shape[:2])
            #                 images.append(img_to_array(img) / 255.0)
            #                 labels.append(int(cls))
            #             except:
            #                 continue

            #     X_final = np.array(images)
            #     y_final = np.array(labels)

            # IMAGE CLASSIFICATION
            if category == "Image Classification":
                images, labels = [], []

                classes = sorted(
                    d for d in os.listdir(user_dir)
                    if os.path.isdir(os.path.join(user_dir, d)) and d.isdigit()
                )

                # ✅ FIXED: build mapping ONCE
                class_to_index = {cls: i for i, cls in enumerate(classes)}

                for cls in classes:
                    cls_dir = os.path.join(user_dir, cls)

                    for img_name in os.listdir(cls_dir):
                        img_path = os.path.join(cls_dir, img_name)
                        try:
                            img = load_img(img_path, target_size=input_shape[:2])
                            images.append(img_to_array(img) / 255.0)
                            labels.append(class_to_index[cls])
                        except:
                            continue

                # ✅ CONVERT FIRST
                X_final = np.array(images, dtype="float32")
                y_final = np.array(labels, dtype="int32")

                # ✅ SHUFFLE SAFELY
                indices = np.random.permutation(len(X_final))
                X_final = X_final[indices]
                y_final = y_final[indices]


            # IMAGE SEGMENTATION
            elif category == "Image Segmentation":
                # imgs, masks = [], []

                # img_dir = os.path.join(user_dir, "images")
                # mask_dir = os.path.join(user_dir, "masks")

                # for name in os.listdir(img_dir):
                #     try:
                #         img = load_img(os.path.join(img_dir, name), target_size=input_shape[:2])
                #         mask = load_img(os.path.join(mask_dir, name), target_size=input_shape[:2], color_mode="grayscale")
                #         imgs.append(img_to_array(img) / 255.0)
                #         mask_arr = img_to_array(mask)
                #         mask_arr = (mask_arr > 0).astype("int32")
                #         masks.append(mask_arr)
                #     except:
                #         continue

                # # X_final = np.array(imgs)
                # # y_final = np.array(masks)
                # X_final = np.array(imgs, dtype="float32")
                # y_final = np.array(masks, dtype="int32")

                # indices = np.random.permutation(len(X_final))
                # X_final = X_final[indices]
                # y_final = y_final[indices]

                imgs, masks = [], []

                img_dir = os.path.join(user_dir, "images")
                mask_dir = os.path.join(user_dir, "masks")

                for name in os.listdir(img_dir):

                    base = os.path.splitext(name)[0]
                    img_path = os.path.join(img_dir, name)

                    mask_path = None
                    for ext in [".png", ".jpg", ".jpeg"]:
                        candidate = os.path.join(mask_dir, base + ext)
                        if os.path.exists(candidate):
                            mask_path = candidate
                            break

                    if mask_path is None:
                        continue

                    img = load_img(img_path, target_size=input_shape[:2])
                    img = img_to_array(img) / 255.0

                    mask = load_img(
                        mask_path,
                        target_size=input_shape[:2],
                        color_mode="grayscale"
                    )

                    mask = img_to_array(mask)
                    mask = np.squeeze(mask)
                    mask = mask.astype("int32") - 1

                    imgs.append(img)
                    masks.append(mask)

                if len(masks) == 0:
                    return jsonify({
                        "error": "No matching image–mask pairs found."
                    }), 400

                X_final = np.array(imgs, dtype="float32")
                y_final = np.array(masks, dtype="int32")

                indices = np.random.permutation(len(X_final))
                X_final = X_final[indices]
                y_final = y_final[indices]

            # IMAGE GENERATION
            # IMAGE GENERATION
            elif category == "Image Generation (Generator Only)":
                with zipfile.ZipFile(dataset_path, "r") as z:
                    names = z.namelist()

                    # ---------------------------
                    # MODE 1: TEXT TO IMAGE
                    # ---------------------------
                    if "text.txt" in names and any(n.startswith("images/") for n in names):
                        # Load Text
                        with z.open("text.txt") as f:
                            text_lines = [line.decode("utf-8").strip() for line in f.readlines() if line.strip()]

                        # Load Images (Sorted to match text lines assumption)
                        img_names = sorted([n for n in names if n.startswith("images/") and not n.endswith("/")])

                        if len(text_lines) != len(img_names):
                            return jsonify({"error": f"Mismatch: {len(text_lines)} text lines vs {len(img_names)} images."}), 400

                        images = []
                        for img_name in img_names:
                            with z.open(img_name) as f:
                                img_data = f.read()
                                img = load_img(BytesIO(img_data), target_size=input_shape[:2])
                                images.append(img_to_array(img) / 255.0)

                        # Tokenize Text
                        tokenizer = Tokenizer(num_words=10000, oov_token="<OOV>")
                        tokenizer.fit_on_texts(text_lines)

                        # Save Tokenizer
                        with open(os.path.join(user_dir, "tokenizer.json"), "w", encoding="utf-8") as f:
                            f.write(tokenizer.to_json())

                        X_final = pad_sequences(tokenizer.texts_to_sequences(text_lines), maxlen=20) # fixed seq len or dynamic?
                        # Let's use a reasonable fixed maxlen or derived from data
                        # For robustness, let's use max(len) but cap at 50
                        max_seq_len = min(max([len(t.split()) for t in text_lines] + [5]), 50)
                        X_final = pad_sequences(tokenizer.texts_to_sequences(text_lines), maxlen=max_seq_len)

                        y_final = np.array(images, dtype="float32")

                    # ---------------------------
                    # MODE 2: IMAGE TO IMAGE
                    # ---------------------------
                    elif any(n.startswith("input/") for n in names) and any(n.startswith("output/") for n in names):
                        input_names = sorted([n for n in names if n.startswith("input/") and not n.endswith("/")])
                        output_names = sorted([n for n in names if n.startswith("output/") and not n.endswith("/")])

                        if len(input_names) != len(output_names):
                            return jsonify({"error": "Mismatch between input/ and output/ folder checks."}), 400

                        X_imgs, y_imgs = [], []

                        # Assuming matching filenames
                        # Create mapping by basename
                        out_map = {os.path.basename(n): n for n in output_names}

                        for inp_name in input_names:
                            base = os.path.basename(inp_name)
                            if base not in out_map:
                                continue

                            with z.open(inp_name) as f:
                                img = load_img(BytesIO(f.read()), target_size=input_shape[:2])
                                X_imgs.append(img_to_array(img) / 255.0)

                            with z.open(out_map[base]) as f:
                                img = load_img(BytesIO(f.read()), target_size=input_shape[:2])
                                y_imgs.append(img_to_array(img) / 255.0)

                        X_final = np.array(X_imgs, dtype="float32")
                        y_final = np.array(y_imgs, dtype="float32")

                    else:
                         return jsonify({"error": "Invalid Image Generation zip structure."}), 400

                indices = np.random.permutation(len(X_final))
                X_final = X_final[indices]
                y_final = y_final[indices]

        if category == "Image Segmentation":
            num_classes = int(np.max(y_final)) + 1

        ok, err = validate_shapes(category, X_final, y_final, input_shape, num_classes)
        if not ok:
            return jsonify({"error": err}), 400

        # -------------------------------------------------
        # SPLIT DATA
        # -------------------------------------------------
        X_train, X_val, y_train, y_val, num_classes, loss_fn = None, None, None, None, None, None
        if category == "Time Series / Audio":
            X_train, X_val, y_train, y_val, num_classes, loss_fn = time_series_preprocess(dataset_path)
        else:
            split = int(0.8 * len(X_final))
            X_train, X_val = X_final[:split], X_final[split:]
            y_train, y_val = y_final[:split], y_final[split:]

        # if num_classes is None and y_final is not None:
        #     num_classes = len(np.unique(y_final))
        # if y_final is not None:
        #     num_classes = int(len(np.unique(y_final)))

        if y_final is not None and category != "Image Segmentation":
            num_classes = int(len(np.unique(y_final)))


        # -------------------------------------------------
        # ARCHITECTURE PREDICTION
        # -------------------------------------------------
        predicted_layers = get_skeleton_from_prompt(
            architecture_model,
            prompt,
            category
        )

        print(predicted_layers)
        if category == "Image Classification":
            for i in range(len(predicted_layers)):
                if i+1 < len(predicted_layers) and predicted_layers[i] == "Dense" and predicted_layers[i+1] == "Softmax":
                    predicted_layers[i] = "Softmax"

        # -------------------------------------------------
        # EVOLUTION + TRAINING
        # -------------------------------------------------
        (model, best_val_acc, best_arch), error = full_pipeline(
            predicted_layers=predicted_layers,
            category=category,
            X_train=X_train,
            y_train=y_train,
            X_val=X_val,
            y_val=y_val,
            input_shape=input_shape,
            num_classes=num_classes
        )

        if error:
            return jsonify({"error": error}), 400

        # -------------------------------------------------
        # SAVE MODEL
        # -------------------------------------------------
        model_path = os.path.join(user_dir, "final_model.keras")
        model.save(model_path)

        inference_data = run_inference(
            category=category,
            model=model,
            X_val=X_val,
            y_val=y_val,
            input_shape=input_shape,
            user_dir=user_dir
        )

        epochs = [i for i in range(20)]
        val_accuracies = [acc for acc in model.history.history["val_accuracy"]] if category != "Time Series / Audio" else [acc for acc in model.history.history["val_loss"]]

        layers = [l.__class__.__name__ for l in model.layers]

        # Return the full architecture
        full_architecture = []
        for layer in model.layers:
            full_architecture.append(layer.get_config())

        return jsonify({
            "status": "success",
            "category": category,
            "architecture": best_arch,
            "full_architecture": full_architecture,
            "val_accuracy": float(best_val_acc),
            "parameters": model.count_params(),
            "download_url": f"/api/download-model/{request_id}",
            "epochs": epochs,
            "val_accuracies": val_accuracies,
            "inference": inference_data,
            "layers": layers
        })

    except Exception as e:
        return jsonify({
            "error": f"Critical System Error: {str(e)}"
        }), 500

# =====================================================
# MODEL DOWNLOAD
# =====================================================
@app.route("/api/download-model/<request_id>", methods=["GET"])
def download_model(request_id):
    path = os.path.join(BASE_UPLOAD_DIR, request_id, "final_model.keras")
    if os.path.exists(path):
        return send_file(path, as_attachment=True)
    return jsonify({"error": "Model not found or expired."}), 404

@app.route("/api/download-tokenizer/<request_id>", methods=["GET"])
def download_tokenizer(request_id):
    path = os.path.join(BASE_UPLOAD_DIR, request_id, "tokenizer.json")
    if os.path.exists(path):
        return send_file(path, as_attachment=True)
    return jsonify({"error": "Tokenizer not found or expired."}), 404

# =====================================================
# MAIN
# =====================================================
if __name__ == "__main__":
    app.run(port=5000, debug=False)
