import zipfile
import os
import pandas as pd
import numpy as np

def validate_dataset(category, file_path):
    ext = os.path.splitext(file_path)[1].lower()

    if category == "Image Classification":
        if ext != ".zip":
            return False, "Image Classification requires a ZIP file."
        with zipfile.ZipFile(file_path) as z:
            folders = {n.split("/")[0] for n in z.namelist() if "/" in n}
            if not folders or not all(f.isdigit() for f in folders):
                return False, "ZIP must contain numeric class folders (0/, 1/, ...)."
        return True, None

    if category == "Image Segmentation":
        if ext != ".zip":
            return False, "Image Segmentation requires a ZIP file."
        with zipfile.ZipFile(file_path) as z:
            names = z.namelist()
            if not any(n.startswith("images/") for n in names):
                return False, "Missing 'images/' directory."
            if not any(n.startswith("masks/") for n in names):
                return False, "Missing 'masks/' directory."
        return True, None

    if category == "Image Generation (Generator Only)":
        if ext != ".zip":
            return False, "Image Generation requires a ZIP file."
        with zipfile.ZipFile(file_path) as z:
            names = z.namelist()

            # Mode 1: Text to Image
            has_text = "text.txt" in names
            has_images_folder = any(n.startswith("images/") for n in names)

            # Mode 2: Image to Image
            has_input_folder = any(n.startswith("input/") for n in names)
            has_output_folder = any(n.startswith("output/") for n in names)

            if has_text and has_images_folder:
                return True, "text_to_image"

            if has_input_folder and has_output_folder:
                return True, "image_to_image"

            return False, "ZIP must contain either ('text.txt' + 'images/') OR ('input/' + 'output/') folders."

    if category == "Natural Language Processing":
        if ext != ".csv":
            return False, "NLP requires a CSV file."
        df = pd.read_csv(file_path)
        if "X" not in df.columns:
            return False, "CSV must contain an 'X' column."
        return True, None

    if category == "Time Series / Audio":
        if ext != ".csv":
            return False, "Time Series requires a CSV file."
        df = pd.read_csv(file_path)
        if "y" not in df.columns:
            return False, "CSV must contain a 'y' column."
        if df.drop(columns=["y"]).select_dtypes(include=np.number).shape[1] == 0:
            return False, "Time Series features must be numeric."
        return True, None

    if category == "Object Detection":
        return False, "Object Detection training is not supported (architecture only)."

    return False, "Unsupported category."
