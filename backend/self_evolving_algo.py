import tensorflow as tf
from tensorflow.keras import layers, Model
from dataclasses import dataclass
from typing import List, Dict

# ======================================================
# CONSTANTS
# ======================================================

CNN_2D = {"Conv2D", "Conv2DTranspose"}
CNN_1D = {"Conv1D"}
RNN_LAYERS = {"SimpleRNN", "LSTM", "GRU", "Bidirectional"}

POOL_2D = {"MaxPooling2D", "AveragePooling2D"}
POOL_1D = {"MaxPooling1D"}

# layers that appear ONCE at the end
HEAD_LAYERS = {
    "Flatten",
    "Dense",
    "Dropout",
    "Softmax",
    "Sigmoid",
    "GlobalAveragePooling2D",
    "GlobalAveragePooling1D",
    "GlobalMaxPooling2D"
}


# ======================================================
# DATA STRUCTURE
# ======================================================

@dataclass
class LayerSpec:
    name: str
    params: Dict


# ======================================================
# SPLIT PREDICTED ARCHITECTURE
# ======================================================

def split_feature_and_head(predicted_layers: List[str]):
    """
    Everything except head layers belongs to the evolving feature block.
    """

    feature_block = []
    head = []

    for layer in predicted_layers:
        if layer == "Input":
            continue

        if layer in HEAD_LAYERS:
            head.append(layer)
        else:
            feature_block.append(layer)

    return feature_block, head


# ======================================================
# BLOCK-WISE EVOLUTION
# ======================================================

def evolve_blocks(block: List[str], depth: int):
    evolved = []
    for _ in range(depth):
        evolved.extend(block)
    return evolved


# ======================================================
# PARAMETER ASSIGNMENT
# ======================================================

def assign_parameters(layer_names: List[str]):
    specs = []
    filters = 32

    for i, layer in enumerate(layer_names):
        params = {}

        if layer in CNN_2D or layer in CNN_1D:
            params = {
                "filters": filters,
                "kernel_size": 3,
                "padding": "same"
            }
            filters = min(filters * 2, 256)

        elif layer in RNN_LAYERS:
            next_is_rnn = (
                i + 1 < len(layer_names)
                and layer_names[i + 1] in RNN_LAYERS
            )
            params = {
                "units": 128,
                "return_sequences": next_is_rnn
            }

        elif layer == "Dense":
            params = {"units": 256}

        elif layer == "Embedding":
            params = {"input_dim": 10000, "output_dim": 128}

        specs.append(LayerSpec(layer, params))

    return specs


# ======================================================
# LAYER FACTORY
# ======================================================

def apply_layer(x, spec: LayerSpec):
    n, p = spec.name, spec.params

    if n == "Conv2D":
        return layers.Conv2D(**p, activation="relu", kernel_initializer='he_normal')(x)

    if n == "Conv1D":
        return layers.Conv1D(**p, activation="relu", kernel_initializer='he_normal')(x)

    if n == "BatchNormalization":
        return layers.BatchNormalization()(x)

    if n == "Dropout":
        return layers.Dropout(0.3)(x)

    if n == "Embedding":
        return layers.Embedding(**p)(x)

    if n == "LSTM":
        return layers.LSTM(**p)(x)

    if n == "GRU":
        return layers.GRU(**p)(x)

    if n == "SimpleRNN":
        return layers.SimpleRNN(**p)(x)

    if n == "Dense":
        return layers.Dense(**p, activation="relu", kernel_initializer='he_normal')(x)

    if n == "MaxPooling2D":
        return layers.MaxPooling2D(pool_size=2)(x)

    if n == "AveragePooling2D":
        return layers.AveragePooling2D(pool_size=2)(x)

    if n == "Conv2DTranspose":
        return layers.Conv2DTranspose(**p, activation="relu", kernel_initializer='he_normal')(x)

    if n == "Conv1DTranspose":
        return layers.Conv1DTranspose(**p, activation="relu", kernel_initializer='he_normal')(x)

    if n == "LayerNormalization":
        return layers.LayerNormalization()(x)

    if n == "BatchNormalization":
        return layers.BatchNormalization()(x)

    # if n == "MaxPooling1D":
    #     return layers.MaxPooling1D(pool_size=2)(x)

    # if n == "AveragePooling1D":
    #     return layers.AveragePooling1D(pool_size=2)(x)

    # if n == "GlobalAveragePooling2D":
    #     return layers.GlobalAveragePooling2D()(x)

    # if n == "GlobalAveragePooling1D":
    #     return layers.GlobalAveragePooling1D()(x)

    # if n == "GlobalMaxPooling2D":
    #     return layers.GlobalMaxPooling2D()(x)

    # if n == "GlobalMaxPooling1D":
    #     return layers.GlobalMaxPooling1D()(x)

    # if n == "Flatten":
    #     return layers.Flatten()(x)

    # if n == "Softmax":
    #     return layers.Softmax()(x)

    # if n == "Sigmoid":
    #     return layers.Sigmoid()(x)

    return x


# ======================================================
# BUILDERS
# ======================================================

def build_image_classification(specs, input_shape, num_classes):
    inp = layers.Input(shape=input_shape)
    x = inp

    for s in specs:
        x = apply_layer(x, s)

    x = layers.GlobalAveragePooling2D()(x)
    x = layers.Dense(128, activation="relu")(x)
    x = layers.Dropout(0.5)(x)
    out = layers.Dense(num_classes, activation="softmax")(x) if num_classes > 1 else layers.Dense(1, activation="sigmoid")(x)

    return Model(inp, out)


def build_segmentation(specs, input_shape, num_classes):
    inp = layers.Input(shape=input_shape)
    x = inp

    for s in specs:
        if s.name in {"Dense", "Flatten"}:
            continue
        x = apply_layer(x, s)

    out = layers.Conv2D(num_classes, 1, activation="softmax")(x)
    return Model(inp, out)


def build_image_to_image(specs, input_shape):
    inp = layers.Input(shape=input_shape)
    x = inp

    for s in specs:
        x = apply_layer(x, s)

    out = layers.Conv2D(input_shape[-1], 1, activation="sigmoid")(x)
    return Model(inp, out)


def build_text_to_image(specs, text_len, img_shape):
    inp = layers.Input(shape=(text_len,))

    # 1. Text Encoder (Fixed Stem)
    x = layers.Embedding(input_dim=10000, output_dim=128)(inp)
    x = layers.LSTM(256)(x)

    # 2. Project and Reshape to start Image Generation
    # Start with 1/8th of the target size (assuming standard usage) or fixed 8x8
    start_dim = max(img_shape[0] // 4, 4)
    x = layers.Dense(start_dim * start_dim * 128)(x)
    x = layers.Reshape((start_dim, start_dim, 128))(x)

    # 3. Apply Evolved Layers (Decoder)
    for s in specs:
        if s.name in {"Flatten", "Dense", "Embedding", "LSTM", "GRU", "SimpleRNN"}:
            continue # Skip incompatible layers in the image generation phase

        # Auto-convert pooling to upsampling for generation if needed
        if "Pooling" in s.name:
             x = layers.UpSampling2D()(x)
        else:
             x = apply_layer(x, s)

    # 4. Final Upsampling to match target size
    # Simple heuristic: Resize to final target
    x = layers.Resizing(img_shape[0], img_shape[1])(x)
    out = layers.Conv2D(img_shape[-1], 1, activation="sigmoid")(x)

    return Model(inp, out)




def build_sequence_model(specs, input_shape, num_classes):
    # inp = layers.Input(shape=input_shape)
    # x = inp

    # for s in specs:
    #     x = apply_layer(x, s)

    # if len(x.shape) == 3:
    #     x = layers.GlobalAveragePooling1D()(x)

    # out = layers.Dense(num_classes, activation="softmax")(x)
    # return Model(inp, out)

    inp = layers.Input(shape=input_shape)
    x = inp

    for i, s in enumerate(specs):
        # 1. Check if we need to pool before a Dense layer
        if s.name == "Dense" and len(x.shape) == 3:
            x = layers.GlobalAveragePooling1D()(x)

        # 2. FIX DOUBLE DENSE: If last layer is Dense, make it the output
        if i == len(specs) - 1 and s.name == "Dense":
            act = 'linear' if num_classes == 1 else 'softmax'
            x = layers.Dense(num_classes, activation=act)(x)
            return Model(inp, x)

        x = apply_layer(x, s)

    # 3. Fallback Output
    if len(x.shape) == 3: x = layers.GlobalAveragePooling1D()(x)
    act = 'linear' if num_classes == 1 else 'softmax'
    out = layers.Dense(num_classes, activation=act)(x)
    return Model(inp, out)

# ======================================================
# MAIN EVOLUTION PIPELINE
# ======================================================

def full_pipeline(
    predicted_layers,
    category,
    X_train, y_train,
    X_val, y_val,
    input_shape,
    num_classes,
    max_depth=3
):
    print("\n================ EVOLUTION STARTED ================\n")

    feature_block, head = split_feature_and_head(predicted_layers)

    if category in {"Natural Language Processing", "Time Series / Audio"}:
        if X_train.ndim == 2:
            print("Auto-reshaping 2D input to 3D for Sequence Model...")
            X_train = tf.expand_dims(X_train, axis=-1)
            X_val = tf.expand_dims(X_val, axis=-1)
            input_shape = (input_shape[0], 1)

        # -----------------------
        # AUTO-FIX: Sanitize Layers (2D -> 1D)
        # -----------------------
        new_feature_block = []
        for l in feature_block:
            if "Conv2D" in l: new_feature_block.append("Conv1D")
            elif "MaxPooling2D" in l: new_feature_block.append("MaxPooling1D")
            elif "GlobalAveragePooling2D" in l: new_feature_block.append("GlobalAveragePooling1D")
            elif "GlobalMaxPooling2D" in l: new_feature_block.append("GlobalMaxPooling1D")
            else: new_feature_block.append(l)
        feature_block = new_feature_block

    if category in {"Image Classification", "Image Segmentation", "Image Generation (Generator Only)"}:
        # If (N, H, W) provided, expand to (N, H, W, 1)
        if hasattr(X_train, 'ndim') and X_train.ndim == 3:
            print("Auto-reshaping 2D Image input (Grayscale) to 3D...")
            X_train = tf.expand_dims(X_train, axis=-1)
            X_val = tf.expand_dims(X_val, axis=-1)
            # update input_shape if it was 2D
            if len(input_shape) == 2:
                input_shape = (input_shape[0], input_shape[1], 1)

        # -----------------------
        # AUTO-FIX: Sanitize Layers (No RNNs in Image)
        # -----------------------
        new_feature_block = []
        for l in feature_block:
            if l in {"LSTM", "GRU", "SimpleRNN", "Bidirectional", "Conv1D", "MaxPooling1D", "GlobalAveragePooling1D"}:
                 # Skip 1D/RNN layers in 2D image task
                 pass
            else:
                 new_feature_block.append(l)
        feature_block = new_feature_block

    if not feature_block:
        return (None, None, None), "No feature layers detected."

    best_model = None
    best_val_acc = -1
    best_arch = None
    has_input = False
    for s in predicted_layers:
        if s == "Input":
            has_input = True
            break

    for depth in range(1, max_depth + 1):

        evolved = evolve_blocks(feature_block, depth)
        architecture = evolved + head

        print(f"\n🔁 DEPTH {depth}")
        print("Architecture:", architecture)

        specs = assign_parameters(architecture)

        try:
            # -----------------------
            # BUILD
            # -----------------------
            if category == "Image Classification":
                model = build_image_classification(specs, input_shape, num_classes)
                if num_classes > 1:
                    loss = "sparse_categorical_crossentropy"
                else:
                    loss = "binary_crossentropy"

            elif category == "Image Segmentation":
                model = build_segmentation(specs, input_shape, num_classes)
                loss = "sparse_categorical_crossentropy"

            elif category == "Image Generation (Generator Only)":
                if len(X_train.shape) == 2:
                    # Text-to-Image Mode
                    model = build_text_to_image(specs, X_train.shape[1], input_shape)
                else:
                    # Image-to-Image Mode
                    model = build_image_to_image(specs, input_shape)
                loss = "mse"

            elif category in {"Natural Language Processing", "Time Series / Audio"}:
                model = build_sequence_model(specs, input_shape, num_classes)
                loss = "mse" if num_classes == 1 else "sparse_categorical_crossentropy"

            else:
                return (None, None, None), "Unsupported category"

            model.compile(
                optimizer=tf.keras.optimizers.Adam(learning_rate=1e-3),
                loss=loss,
                metrics=["mae"] if num_classes == 1 else ["accuracy"]
            )

            history = model.fit(
                X_train, y_train,
                validation_data=(X_val, y_val),
                epochs=3,
                batch_size=32,
                verbose=0
            )

            if num_classes == 1:
                val_mae = history.history.get("val_mae", [1e6])[-1]
                val_acc = 1 / (1 + val_mae)
            else:
                val_acc = max(history.history.get("val_accuracy", [0]))

            print(f"✅ val_accuracy = {val_acc:.4f}")
            print("Used layers:",
                  [l.__class__.__name__ for l in model.layers])

            if val_acc > best_val_acc:
                best_val_acc = val_acc
                best_model = model
                best_arch = ["Input"] + architecture if has_input else architecture

        except Exception as e:
            print(f"⚠️ rejected: {e}")
            continue

    if best_model is None:
        return (None, None, None), "Evolution failed."

    print("\n🏆 BEST ARCHITECTURE:")
    print(best_arch)

    # -----------------------
    # FINAL TRAIN
    # -----------------------
    best_model.fit(
        X_train, y_train,
        validation_data=(X_val, y_val),
        epochs=10,
        batch_size=32,
        verbose=1
    )

    return (best_model, best_val_acc, best_arch), None









# import tensorflow as tf
# from tensorflow.keras import layers, Model
# from dataclasses import dataclass
# from typing import List, Dict, Tuple
# import numpy as np

# # =====================================================
# # CONSTANTS
# # =====================================================

# CNN_2D = {"Conv2D", "Conv2DTranspose"}
# CNN_1D = {"Conv1D"}
# RNN_LAYERS = {"SimpleRNN", "LSTM", "GRU", "Bidirectional"}
# POOL_2D = {"MaxPooling2D", "AveragePooling2D"}
# POOL_1D = {"MaxPooling1D"}
# FLATTEN_LAYERS = {
#     "Flatten", "GlobalAveragePooling2D", "GlobalMaxPooling2D",
#     "GlobalAveragePooling1D"
# }

# NON_REPEATABLE = {
#     "Embedding", "Flatten",
#     "GlobalAveragePooling2D",
#     "GlobalMaxPooling2D",
#     "GlobalAveragePooling1D"
# }

# # =====================================================
# # DATA STRUCTURE
# # =====================================================

# @dataclass
# class LayerSpec:
#     name: str
#     params: Dict

# # =====================================================
# # EVOLUTION CORE
# # =====================================================

# def expand_architecture(base_layers: List[str], depth: int):
#     expanded = []
#     for layer in base_layers:
#         expanded.append(layer)
#         if layer not in NON_REPEATABLE:
#             for _ in range(depth - 1):
#                 expanded.append(layer)
#     return expanded


# def assign_parameters(layers_list: List[str]):
#     specs = []
#     filters = 32

#     for i, layer in enumerate(layers_list):
#         params = {}

#         if layer in CNN_2D or layer in CNN_1D:
#             params = {"filters": filters, "kernel_size": 3, "padding": "same"}
#             filters = min(filters * 2, 256)

#         elif layer in RNN_LAYERS:
#             next_is_rnn = i + 1 < len(layers_list) and layers_list[i + 1] in RNN_LAYERS
#             params = {"units": 128, "return_sequences": next_is_rnn}

#         elif layer == "Dense":
#             params = {"units": 256}

#         elif layer == "Embedding":
#             params = {"input_dim": 10000, "output_dim": 128}

#         specs.append(LayerSpec(layer, params))

#     return specs

# # =====================================================
# # LAYER FACTORY
# # =====================================================

# def apply_layer(x, spec: LayerSpec):
#     n, p = spec.name, spec.params

#     if n == "Conv2D":
#         return layers.Conv2D(**p, activation="relu")(x)
#     if n == "Conv1D":
#         return layers.Conv1D(**p, activation="relu")(x)
#     if n == "BatchNormalization":
#         return layers.BatchNormalization()(x)
#     if n == "Dropout":
#         return layers.Dropout(0.3)(x)
#     if n == "Embedding":
#         return layers.Embedding(**p)(x)
#     if n == "LSTM":
#         return layers.LSTM(**p)(x)
#     if n == "GRU":
#         return layers.GRU(**p)(x)
#     if n == "SimpleRNN":
#         return layers.SimpleRNN(**p)(x)
#     if n == "Dense":
#         return layers.Dense(**p, activation="relu")(x)

#     return x

# # =====================================================
# # CATEGORY BUILDERS
# # =====================================================

# def build_image_classification(specs, input_shape, num_classes):
#     inp = layers.Input(shape=input_shape)
#     x = inp

#     for s in specs:
#         x = apply_layer(x, s)

#     x = layers.GlobalAveragePooling2D()(x)
#     out = layers.Dense(num_classes, activation="softmax")(x)
#     return Model(inp, out)


# def build_segmentation(specs, input_shape, num_classes):
#     inp = layers.Input(shape=input_shape)
#     x = inp

#     for s in specs:
#         if s.name in {"Dense", "Flatten"}:
#             continue
#         x = apply_layer(x, s)

#     out = layers.Conv2D(num_classes, 1, activation="softmax")(x)
#     return Model(inp, out)


# def build_image_generation(specs, input_shape):
#     inp = layers.Input(shape=input_shape)
#     x = inp

#     for s in specs:
#         x = apply_layer(x, s)

#     out = layers.Conv2D(input_shape[-1], 1, activation="sigmoid")(x)
#     return Model(inp, out)

# def build_nlp(specs, input_shape, num_classes):
#     inp = layers.Input(shape=input_shape)
#     x = inp

#     for s in specs:
#         x = apply_layer(x, s)

#     # ✅ FIX
#     if len(x.shape) == 3:
#         x = layers.GlobalAveragePooling1D()(x)

#     out = layers.Dense(num_classes, activation="softmax")(x)
#     return Model(inp, out)

# def build_time_series(specs, input_shape, num_classes):
#     return build_nlp(specs, input_shape, num_classes)

# # =====================================================
# # TRUE SELF-EVOLVING PIPELINE
# # =====================================================

# def full_pipeline(
#     predicted_layers,
#     category,
#     X_train, y_train,
#     X_val, y_val,
#     input_shape,
#     num_classes,
#     max_depth=3
# ):
#     """
#     Tries multiple evolved architectures and selects the best.
#     """

#     best_model = None
#     best_score = -1
#     best_arch = None

#     for depth in range(1, max_depth + 1):

#         expanded = expand_architecture(predicted_layers, depth)
#         print(f"\n🔁 Testing depth = {depth}")
#         print("Architecture:", expanded)

#         specs = assign_parameters(expanded)

#         try:
#             # -----------------------------
#             # BUILD MODEL
#             # -----------------------------
#             if category == "Image Classification":
#                 model = build_image_classification(specs, input_shape, num_classes)
#                 loss = "sparse_categorical_crossentropy"

#             elif category == "Image Segmentation":
#                 model = build_segmentation(specs, input_shape, num_classes)
#                 loss = "sparse_categorical_crossentropy"

#             elif category == "Image Generation (Generator Only)":
#                 model = build_image_generation(specs, input_shape)
#                 loss = "mse"

#             elif category == "Natural Language Processing":
#                 model = build_nlp(specs, input_shape, num_classes)
#                 loss = "sparse_categorical_crossentropy"

#             elif category == "Time Series / Audio":
#                 model = build_time_series(specs, input_shape, num_classes)
#                 loss = "sparse_categorical_crossentropy"

#             else:
#                 return (None, None, None), "Unsupported category"

#             model.compile(
#                 optimizer="adam",
#                 loss=loss,
#                 metrics=["accuracy"]
#             )

#             # -----------------------------
#             # TRAIN
#             # -----------------------------
#             history = model.fit(
#                 X_train, y_train,
#                 validation_data=(X_val, y_val),
#                 epochs=3,
#                 batch_size=32,
#                 verbose=0
#             )

#             score = max(history.history.get("val_accuracy", [0]))

#             print(f"Depth {depth} → val_acc={score:.4f}")

#             # -----------------------------
#             # SELECTION
#             # -----------------------------
#             if score > best_score:
#                 best_score = score
#                 best_model = model
#                 best_arch = expanded

#         except Exception as e:
#             print(f"⚠️ Evolution failed at depth {depth}: {e}")
#             continue

#     if best_model is None:
#         return (None, None, None), "Evolution failed for all architectures."

#     print("✅ Selected architecture:")
#     print(best_arch)

#     best_model.fit(
#         X_train, y_train,
#         validation_data=(X_val, y_val),
#         epochs=20,
#         batch_size=32,
#         verbose=1
#     )

#     return (best_model, best_score, best_arch), None





# import tensorflow as tf
# from tensorflow.keras import layers, Model
# from dataclasses import dataclass
# from typing import List, Dict, Tuple
# import numpy as np

# # =====================================================
# # CONSTANTS
# # =====================================================

# CNN_2D = {"Conv2D", "Conv2DTranspose"}
# CNN_1D = {"Conv1D"}
# RNN_LAYERS = {"SimpleRNN", "LSTM", "GRU", "Bidirectional"}
# POOL_2D = {"MaxPooling2D", "AveragePooling2D"}
# POOL_1D = {"MaxPooling1D"}
# FLATTEN_LAYERS = {
#     "Flatten", "GlobalAveragePooling2D", "GlobalMaxPooling2D",
#     "GlobalAveragePooling1D"
# }

# NON_REPEATABLE = {
#     "Embedding", "Flatten",
#     "GlobalAveragePooling2D",
#     "GlobalMaxPooling2D",
#     "GlobalAveragePooling1D"
# }

# # =====================================================
# # DATA STRUCTURE
# # =====================================================

# @dataclass
# class LayerSpec:
#     name: str
#     params: Dict

# # =====================================================
# # EVOLUTION CORE
# # =====================================================

# def expand_architecture(base_layers: List[str], depth: int):
#     expanded = []
#     for layer in base_layers:
#         expanded.append(layer)
#         if layer not in NON_REPEATABLE:
#             for _ in range(depth - 1):
#                 expanded.append(layer)
#     return expanded


# def assign_parameters(layers_list: List[str]):
#     specs = []
#     filters = 32

#     for i, layer in enumerate(layers_list):
#         params = {}

#         if layer in CNN_2D or layer in CNN_1D:
#             params = {"filters": filters, "kernel_size": 3, "padding": "same"}
#             filters = min(filters * 2, 256)

#         elif layer in RNN_LAYERS:
#             next_is_rnn = i + 1 < len(layers_list) and layers_list[i + 1] in RNN_LAYERS
#             params = {"units": 128, "return_sequences": next_is_rnn}

#         elif layer == "Dense":
#             params = {"units": 256}

#         elif layer == "Embedding":
#             params = {"input_dim": 10000, "output_dim": 128}

#         specs.append(LayerSpec(layer, params))

#     return specs

# # =====================================================
# # LAYER FACTORY
# # =====================================================

# def apply_layer(x, spec: LayerSpec):
#     n, p = spec.name, spec.params

#     if n == "Conv2D":
#         return layers.Conv2D(**p, activation="relu")(x)
#     if n == "Conv1D":
#         return layers.Conv1D(**p, activation="relu")(x)
#     if n == "BatchNormalization":
#         return layers.BatchNormalization()(x)
#     if n == "Dropout":
#         return layers.Dropout(0.3)(x)
#     if n == "Embedding":
#         return layers.Embedding(**p)(x)
#     if n == "LSTM":
#         return layers.LSTM(**p)(x)
#     if n == "GRU":
#         return layers.GRU(**p)(x)
#     if n == "SimpleRNN":
#         return layers.SimpleRNN(**p)(x)
#     if n == "Dense":
#         return layers.Dense(**p, activation="relu")(x)

#     return x

# # =====================================================
# # CATEGORY BUILDERS
# # =====================================================

# def build_image_classification(specs, input_shape, num_classes):
#     inp = layers.Input(shape=input_shape)
#     x = inp

#     for s in specs:
#         x = apply_layer(x, s)

#     x = layers.GlobalAveragePooling2D()(x)
#     out = layers.Dense(num_classes, activation="softmax")(x)
#     return Model(inp, out)


# def build_segmentation(specs, input_shape, num_classes):
#     inp = layers.Input(shape=input_shape)
#     x = inp

#     for s in specs:
#         if s.name in {"Dense", "Flatten"}:
#             continue
#         x = apply_layer(x, s)

#     out = layers.Conv2D(num_classes, 1, activation="softmax")(x)
#     return Model(inp, out)


# def build_image_generation(specs, input_shape):
#     inp = layers.Input(shape=input_shape)
#     x = inp

#     for s in specs:
#         x = apply_layer(x, s)

#     out = layers.Conv2D(input_shape[-1], 1, activation="sigmoid")(x)
#     return Model(inp, out)

# def build_nlp(specs, input_shape, num_classes):
#     inp = layers.Input(shape=input_shape)
#     x = inp

#     for s in specs:
#         x = apply_layer(x, s)

#     # ✅ FIX
#     if len(x.shape) == 3:
#         x = layers.GlobalAveragePooling1D()(x)

#     out = layers.Dense(num_classes, activation="softmax")(x)
#     return Model(inp, out)

# def build_time_series(specs, input_shape, num_classes):
#     return build_nlp(specs, input_shape, num_classes)

# # =====================================================
# # TRUE SELF-EVOLVING PIPELINE
# # =====================================================

# def full_pipeline(
#     predicted_layers,
#     category,
#     X_train, y_train,
#     X_val, y_val,
#     input_shape,
#     num_classes,
#     max_depth=3
# ):
#     """
#     Tries multiple evolved architectures and selects the best.
#     """

#     best_model = None
#     best_score = -1
#     best_arch = None

#     for depth in range(1, max_depth + 1):

#         expanded = expand_architecture(predicted_layers, depth)
#         print(f"\n🔁 Testing depth = {depth}")
#         print("Architecture:", expanded)

#         specs = assign_parameters(expanded)

#         try:
#             # -----------------------------
#             # BUILD MODEL
#             # -----------------------------
#             if category == "Image Classification":
#                 model = build_image_classification(specs, input_shape, num_classes)
#                 loss = "sparse_categorical_crossentropy"

#             elif category == "Image Segmentation":
#                 model = build_segmentation(specs, input_shape, num_classes)
#                 loss = "sparse_categorical_crossentropy"

#             elif category == "Image Generation (Generator Only)":
#                 model = build_image_generation(specs, input_shape)
#                 loss = "mse"

#             elif category == "Natural Language Processing":
#                 model = build_nlp(specs, input_shape, num_classes)
#                 loss = "sparse_categorical_crossentropy"

#             elif category == "Time Series / Audio":
#                 model = build_time_series(specs, input_shape, num_classes)
#                 loss = "sparse_categorical_crossentropy"

#             else:
#                 return (None, None), "Unsupported category"

#             model.compile(
#                 optimizer="adam",
#                 loss=loss,
#                 metrics=["accuracy"]
#             )

#             # -----------------------------
#             # TRAIN
#             # -----------------------------
#             history = model.fit(
#                 X_train, y_train,
#                 validation_data=(X_val, y_val),
#                 epochs=3,
#                 batch_size=32,
#                 verbose=0
#             )

#             score = max(history.history.get("val_accuracy", [0]))

#             print(f"Depth {depth} → val_acc={score:.4f}")

#             # -----------------------------
#             # SELECTION
#             # -----------------------------
#             if score > best_score:
#                 best_score = score
#                 best_model = model
#                 best_arch = expanded

#         except Exception as e:
#             print(f"⚠️ Evolution failed at depth {depth}: {e}")
#             continue

#     if best_model is None:
#         return (None, None, None), "Evolution failed for all architectures."

#     print("✅ Selected architecture:")
#     print(best_arch)

#     best_model.fit(
#         X_train, y_train,
#         validation_data=(X_val, y_val),
#         epochs=20,
#         batch_size=32,
#         verbose=1
#     )

#     return (best_model, best_score, best_arch), None





# import tensorflow as tf
# from tensorflow.keras import layers, Model
# from dataclasses import dataclass
# from typing import List, Dict, Tuple

# # ======================================================
# # CONSTANTS
# # ======================================================

# CNN_2D = {"Conv2D", "Conv2DTranspose"}
# CNN_1D = {"Conv1D"}
# RNN_LAYERS = {"SimpleRNN", "LSTM", "GRU", "Bidirectional"}

# POOL_2D = {"MaxPooling2D", "AveragePooling2D"}
# POOL_1D = {"MaxPooling1D"}

# # FEATURE_LAYERS = (
# #     CNN_2D | CNN_1D | RNN_LAYERS |
# #     POOL_2D | POOL_1D |
# #     {"BatchNormalization", "Dropout"}
# # )

# FEATURE_LAYERS = (
#     CNN_2D |
#     CNN_1D |
#     RNN_LAYERS |
#     POOL_2D |
#     POOL_1D |
#     {
#         "BatchNormalization",
#         "LayerNormalization",
#         "Dropout",
#         "Embedding",
#         "MultiHeadAttention"
#     }
# )

# HEAD_LAYERS = {
#     # "Flatten",
#     "Dense",
#     "Softmax",
#     "Sigmoid",
#     # "GlobalAveragePooling2D",
#     # "GlobalAveragePooling1D",
#     # "GlobalMaxPooling2D"
# }

# NON_REPEATABLE = {
#     "Flatten",
#     "Embedding",
#     "GlobalAveragePooling2D",
#     "GlobalAveragePooling1D",
#     "GlobalMaxPooling2D"
# }

# # ======================================================
# # DATA STRUCTURE
# # ======================================================

# @dataclass
# class LayerSpec:
#     name: str
#     params: Dict


# # ======================================================
# # SPLIT BACKBONE & HEAD
# # ======================================================

# def split_backbone_head(predicted_layers: List[str]):
#     backbone = []
#     head = []

#     for layer in predicted_layers:
#         if layer in FEATURE_LAYERS:
#             backbone.append(layer)
#         elif layer in HEAD_LAYERS:
#             head.append(layer)

#     return backbone, head


# # ======================================================
# # BLOCK-WISE EVOLUTION
# # ======================================================

# def evolve_backbone(backbone: List[str], depth: int):
#     evolved = []
#     for _ in range(depth):
#         evolved.extend(backbone)
#     return evolved


# # ======================================================
# # PARAMETER ASSIGNMENT
# # ======================================================

# def assign_parameters(layers_list: List[str]):
#     specs = []
#     filters = 32

#     for i, layer in enumerate(layers_list):
#         params = {}

#         if layer in CNN_2D or layer in CNN_1D:
#             params = {
#                 "filters": filters,
#                 "kernel_size": 3,
#                 "padding": "same"
#             }
#             filters = min(filters * 2, 256)

#         # elif layer in RNN_LAYERS:
#         #     next_is_rnn = (
#         #         i + 1 < len(layers_list)
#         #         and layers_list[i + 1] in RNN_LAYERS
#         #     )
#         #     params = {
#         #         "units": 128,
#         #         "return_sequences": next_is_rnn
#         #     }

#         elif layer in RNN_LAYERS:
#             future_has_rnn = any(
#                 l in RNN_LAYERS
#                 for l in layers_list[i + 1:]
#             )

#             params = {
#                 "units": 128,
#                 "return_sequences": future_has_rnn
#             }

#         elif layer == "Dense":
#             params = {"units": 256}

#         elif layer == "Embedding":
#             params = {"input_dim": 10000, "output_dim": 128}

#         specs.append(LayerSpec(layer, params))

#     return specs


# # ======================================================
# # SAFE LAYER FACTORY
# # ======================================================

# def apply_layer(x, spec: LayerSpec):
#     n, p = spec.name, spec.params

#     if n == "Conv2D":
#         return layers.Conv2D(**p, activation="relu")(x)
#     if n == "Conv1D":
#         return layers.Conv1D(**p, activation="relu")(x)
#     if n == "BatchNormalization":
#         return layers.BatchNormalization()(x)
#     if n == "Dropout":
#         return layers.Dropout(0.3)(x)
#     if n == "Embedding":
#         return layers.Embedding(**p)(x)
#     if n == "LSTM":
#         return layers.LSTM(**p)(x)
#     if n == "GRU":
#         return layers.GRU(**p)(x)
#     if n == "SimpleRNN":
#         return layers.SimpleRNN(**p)(x)
#     if n == "Dense":
#         return layers.Dense(**p, activation="relu")(x)

#     return x


# # ======================================================
# # CATEGORY BUILDERS
# # ======================================================

# def build_image_classification(specs, input_shape, num_classes):
#     inp = layers.Input(shape=input_shape)
#     x = inp

#     for s in specs:
#         x = apply_layer(x, s)

#     x = layers.GlobalAveragePooling2D()(x)
#     out = layers.Dense(num_classes, activation="softmax")(x)

#     return Model(inp, out)


# def build_segmentation(specs, input_shape, num_classes):
#     inp = layers.Input(shape=input_shape)
#     x = inp

#     for s in specs:
#         if s.name in {"Dense", "Flatten"}:
#             continue
#         x = apply_layer(x, s)

#     out = layers.Conv2D(num_classes, 1, activation="softmax")(x)
#     return Model(inp, out)


# def build_image_generation(specs, input_shape):
#     inp = layers.Input(shape=input_shape)
#     x = inp

#     for s in specs:
#         x = apply_layer(x, s)

#     out = layers.Conv2D(input_shape[-1], 1, activation="sigmoid")(x)
#     return Model(inp, out)

# def transformer_block(
#     x,
#     num_heads=4,
#     key_dim=64,
#     ff_dim=256,
#     dropout=0.1
# ):
#     d_model = x.shape[-1]  # ← THIS IS THE KEY

#     # ------------------------------
#     # Multi-head self-attention
#     # ------------------------------
#     attn_output = layers.MultiHeadAttention(
#         num_heads=num_heads,
#         key_dim=key_dim
#     )(x, x)

#     # 🔥 PROJECT BACK TO d_model
#     attn_output = layers.Dense(d_model)(attn_output)

#     # Residual 1
#     x = layers.Add()([x, attn_output])
#     x = layers.LayerNormalization()(x)

#     # ------------------------------
#     # Feed-forward network
#     # ------------------------------
#     ffn = layers.Dense(ff_dim, activation="relu")(x)
#     ffn = layers.Dense(d_model)(ffn)

#     # Residual 2
#     x = layers.Add()([x, ffn])
#     x = layers.LayerNormalization()(x)

#     return x

# def build_sequence_model(
#     specs,
#     input_shape,
#     num_classes,
#     category
# ):
#     inp = layers.Input(shape=input_shape)
#     x = inp

#     for s in specs:
#         if s.name in {
#             "GlobalAveragePooling1D",
#             "GlobalAveragePooling2D",
#             "GlobalMaxPooling2D",
#             "Flatten"
#         }:
#             continue
#         if s.name == "Embedding":
#             if category == "Natural Language Processing":
#                 x = layers.Embedding(**s.params)(x)
#             else:
#                 # Skip embedding for numeric signals
#                 continue

#         elif s.name == "MultiHeadAttention":
#             # Ensure 3D tensor
#             if len(x.shape) != 3:
#                 # raise ValueError(
#                 #     "MultiHeadAttention requires 3D tensor"
#                 # )
#                 continue
#             x = transformer_block(x)

#         elif s.name in {"LSTM", "GRU", "SimpleRNN"}:
#             x = getattr(layers, s.name)(**s.params)(x)

#         else:
#             x = apply_layer(x, s)

#     if len(x.shape) == 3:
#         x = layers.GlobalAveragePooling1D()(x)

#     out = layers.Dense(num_classes, activation="softmax")(x)

#     return Model(inp, out)

# # def build_nlp(specs, input_shape, num_classes):
# #     inp = layers.Input(shape=input_shape)
# #     x = inp

# #     for s in specs:
# #         x = apply_layer(x, s)

# #     if x.shape.rank == 3:
# #         x = layers.GlobalAveragePooling1D()(x)

# #     out = layers.Dense(num_classes, activation="softmax")(x)
# #     return Model(inp, out)


# # def build_time_series(specs, input_shape, num_classes):
# #     return build_nlp(specs, input_shape, num_classes)


# # ======================================================
# # TRUE SELF-EVOLVING PIPELINE
# # ======================================================

# def full_pipeline(
#     predicted_layers,
#     category,
#     X_train, y_train,
#     X_val, y_val,
#     input_shape,
#     num_classes,
#     max_depth=4
# ):
#     print("\n================ EVOLUTION STARTED ================\n")

#     predicted_layers = [
#         l for l in predicted_layers
#         if l not in {"Input"}
#     ]

#     if category == "Natural Language Processing":
#         predicted_layers = [
#             l for l in predicted_layers
#             if l not in {
#                 "GlobalAveragePooling1D",
#                 "GlobalAveragePooling2D",
#                 "GlobalMaxPooling2D",
#                 "Flatten"
#             }
#         ]

#     backbone, head = split_backbone_head(predicted_layers)

#     if not backbone:
#         return (None, None), "No valid backbone layers found."

#     best_model = None
#     best_val_acc = -1
#     best_arch = None

#     for depth in range(1, max_depth + 1):

#         evolved_backbone = evolve_backbone(backbone, depth)
#         architecture = evolved_backbone + head

#         print(f"\n🔁 DEPTH {depth}")
#         print("Architecture:", architecture)

#         specs = assign_parameters(architecture)

#         try:
#             # ----------------------------------
#             # BUILD
#             # ----------------------------------
#             if category == "Image Classification":
#                 model = build_image_classification(specs, input_shape, num_classes)
#                 loss = "sparse_categorical_crossentropy"

#             elif category == "Image Segmentation":
#                 model = build_segmentation(specs, input_shape, num_classes)
#                 loss = "sparse_categorical_crossentropy"

#             elif category == "Image Generation (Generator Only)":
#                 model = build_image_generation(specs, input_shape)
#                 loss = "mse"

#             elif category == "Natural Language Processing":
#                 model = build_sequence_model(specs, input_shape, num_classes, "Natural Language Processing")
#                 loss = "sparse_categorical_crossentropy"

#             elif category == "Time Series / Audio":
#                 model = build_sequence_model(specs, input_shape, num_classes, "Time Series / Audio")
#                 loss = "sparse_categorical_crossentropy"

#             else:
#                 return (None, None), "Unsupported category."

#             model.compile(
#                 optimizer="adam",
#                 loss=loss,
#                 metrics=["accuracy"]
#             )

#             history = model.fit(
#                 X_train, y_train,
#                 validation_data=(X_val, y_val),
#                 epochs=3,
#                 batch_size=32,
#                 verbose=0
#             )

#             val_acc = max(history.history.get("val_accuracy", [0]))

#             print(f"✅ val_accuracy = {val_acc:.4f}")

#             if val_acc > best_val_acc:
#                 best_val_acc = val_acc
#                 best_model = model
#                 best_arch = architecture

#         except Exception as e:
#             print(f"⚠️ Evolution failed: {e}")
#             continue

#     if best_model is None:
#         return (None, None), "Evolution failed for all architectures."

#     print("\n🏆 BEST ARCHITECTURE SELECTED:")
#     print(best_arch)

#     # ----------------------------------
#     # FINAL TRAINING
#     # ----------------------------------
#     best_model.fit(
#         X_train, y_train,
#         validation_data=(X_val, y_val),
#         epochs=20,
#         batch_size=32,
#         verbose=1
#     )

#     return (best_model, best_val_acc, best_arch), None






# import tensorflow as tf
# from tensorflow.keras import layers, Model
# from dataclasses import dataclass
# from typing import List, Dict, Tuple

# # =====================================================
# # DATA STRUCTURES
# # =====================================================

# @dataclass
# class LayerSpec:
#     name: str
#     params: Dict

# # =====================================================
# # EVOLUTION UTILITIES (SHARED)
# # =====================================================

# NON_REPEATABLE = {"Embedding", "Flatten"}

# def expand_architecture(layers_list, depth):
#     expanded = []
#     for l in layers_list:
#         expanded.append(l)
#         if l not in NON_REPEATABLE:
#             for _ in range(depth - 1):
#                 expanded.append(l)
#     return expanded

# def assign_parameters(layers_list):
#     specs = []
#     for i, name in enumerate(layers_list):
#         p = {}
#         if name in {"Conv2D", "Conv1D"}:
#             p = {"filters": 64, "kernel_size": 3, "padding": "same"}
#         elif name in {"LSTM", "GRU", "SimpleRNN"}:
#             p = {"units": 128, "return_sequences": True}
#         elif name == "Dense":
#             p = {"units": 256}
#         elif name == "Embedding":
#             p = {"input_dim": 10000, "output_dim": 128}
#         specs.append(LayerSpec(name, p))
#     return specs

# # =====================================================
# # LAYER FACTORY (SAFE)
# # =====================================================

# def apply_layer(x, spec: LayerSpec):
#     n, p = spec.name, spec.params

#     if n == "Conv2D":
#         return layers.Conv2D(**p, activation="relu")(x)
#     if n == "Conv1D":
#         return layers.Conv1D(**p, activation="relu")(x)
#     if n == "BatchNormalization":
#         return layers.BatchNormalization()(x)
#     if n == "Dropout":
#         return layers.Dropout(0.3)(x)
#     if n == "Embedding":
#         return layers.Embedding(**p)(x)
#     if n == "LSTM":
#         return layers.LSTM(**p)(x)
#     if n == "GRU":
#         return layers.GRU(**p)(x)
#     if n == "SimpleRNN":
#         return layers.SimpleRNN(**p)(x)
#     if n == "Dense":
#         return layers.Dense(**p, activation="relu")(x)

#     return x

# # =====================================================
# # CATEGORY-SPECIFIC PIPELINES
# # =====================================================

# def evolve_image_classification(specs, input_shape, num_classes):
#     inp = layers.Input(shape=input_shape)
#     x = inp
#     for s in specs:
#         x = apply_layer(x, s)
#     x = layers.GlobalAveragePooling2D()(x)
#     out = layers.Dense(num_classes, activation="softmax")(x)
#     return Model(inp, out)

# def evolve_segmentation(specs, input_shape, num_classes):
#     inp = layers.Input(shape=input_shape)
#     x = inp
#     for s in specs:
#         if s.name in {"Dense", "Flatten"}:
#             continue
#         x = apply_layer(x, s)
#     out = layers.Conv2D(num_classes, (1, 1), activation="softmax")(x)
#     return Model(inp, out)

# def evolve_image_generation(specs, input_shape):
#     inp = layers.Input(shape=input_shape)
#     x = inp
#     for s in specs:
#         x = apply_layer(x, s)
#     out = layers.Conv2D(input_shape[-1], (1, 1), activation="sigmoid")(x)
#     return Model(inp, out)

# def evolve_nlp(specs, input_shape, num_classes):
#     inp = layers.Input(shape=input_shape)
#     x = inp
#     for s in specs:
#         x = apply_layer(x, s)
#     x = layers.GlobalAveragePooling1D()(x)
#     out = layers.Dense(num_classes, activation="softmax")(x)
#     return Model(inp, out)

# def evolve_time_series(specs, input_shape, num_classes):
#     return evolve_nlp(specs, input_shape, num_classes)

# # =====================================================
# # MASTER DISPATCHER
# # =====================================================

# def full_pipeline(predicted_layers, category,
#                   X_train, y_train, X_val, y_val,
#                   input_shape, num_classes, depth_factor=2):

#     expanded = expand_architecture(predicted_layers, depth_factor)
#     specs = assign_parameters(expanded)

#     if category == "Image Classification":
#         model = evolve_image_classification(specs, input_shape, num_classes)
#         loss = "sparse_categorical_crossentropy"

#     elif category == "Image Segmentation":
#         model = evolve_segmentation(specs, input_shape, num_classes)
#         loss = "sparse_categorical_crossentropy"

#     elif category == "Image Generation (Generator Only)":
#         model = evolve_image_generation(specs, input_shape)
#         loss = "mse"

#     elif category == "Natural Language Processing":
#         model = evolve_nlp(specs, input_shape, num_classes)
#         loss = "sparse_categorical_crossentropy"

#     elif category == "Time Series / Audio":
#         model = evolve_time_series(specs, input_shape, num_classes)
#         loss = "sparse_categorical_crossentropy"

#     elif category == "Object Detection":
#         return (None, None), "Object Detection training not supported."

#     else:
#         return (None, None), "Unsupported category."

#     model.compile(optimizer="adam", loss=loss, metrics=["accuracy"])

#     history = model.fit(
#         X_train, y_train,
#         validation_data=(X_val, y_val),
#         epochs=3,
#         batch_size=32
#     )

#     return (model, history), None
