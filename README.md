# Hybrid Self-Evolving Neural Network (H-SENN)

![Project Status](https://img.shields.io/badge/Status-Active-success)
![Python](https://img.shields.io/badge/Python-3.8%2B-blue)
![React](https://img.shields.io/badge/Frontend-React%20%7C%20Vite-61DAFB)
![TensorFlow](https://img.shields.io/badge/Core-TensorFlow%20%7C%20Keras-orange)

## 🧠 Overview

**Hybrid Self-Evolving Neural Network (H-SENN)** is an advanced AI system capable of **automatically designing, evolving, and training deep learning architectures** tailored to specific user-defined tasks.

Unlike traditional AutoML which often relies on brute-force search, H-SENN uses a **Language-Model-Driven Generator** to create an initial architectural "skeleton" from a natural language prompt, and then employs a **Genetic Evolutionary Algorithm** to optimize the depth, layer composition, and hyperparameters (filters, units, kernels).

Whether you need a **CNN** for image classification, a **UNet-like** structure for segmentation, or an **LSTM/GRU** block for time-series forecasting, H-SENN evolves the optimal model for you.

---

## ✨ Key Features

-   **🧬 Biological-Inspired Evolution**: Starts with a prompt-based "seed" architecture and evolves it through multiple generations (depths) to find the best performing structure.
-   **🗣️ Natural Language Interface**: Just describe your problem (e.g., *"I want to classify different types of flowers"*), and the system infers the task category and base layers.
-   **🔀 Hybrid Architecture Support**: Seamlessly mixes and matches:
    -   **Conv2D / Pooling** (Computer Vision)
    -   **LSTM / GRU / RNN** (Sequential Data)
    -   **Dense / Dropout** (Tabular/General)
-   **📊 Multi-Modal Support**:
    -   📸 **Image Classification** (Multi-class & Binary)
    -   🖌️ **Image Segmentation** (Mask generation)
    -   🎨 **Image Generation** (Text-to-Image & Image-to-Image)
    -   📝 **Natural Language Processing (NLP)** (Text classification)
    -   📈 **Time Series / Audio** (Forecasting & Analysis)
-   **🛡️ Robust Data Validation**: Strict, automated checks ensure your datasets are correctly formatted before training begins.
-   **⚡ Real-Time Visualization**: Watch the evolution process, view validation metrics, and inspect the final architecture in a modern React UI.

---

## 🛠️ Architecture

The system consists of two main components:

1.  **Frontend (React + Vite)**: A modern, responsive dashboard to upload data, configure parameters, and visualize training results.
2.  **Backend (Flask + TensorFlow)**:
    -   **Controller**: Handles API requests and data preprocessing.
    -   **Evolution Engine**: Runs the genetic loop (Population Generation → Evaluation → Selection).
    -   **Model Builder**: Dynamically constructs Keras models from abstract layer specifications.

---

## 🚀 Installation

### Prerequisites
-   **Python 3.8+**
-   **Node.js 16+** & **npm**

### 1. Backend Setup

```bash
# Clone the repository
git clone https://github.com/yourusername/Hybrid-Self-Evolving-Neural-Network.git
cd Hybrid-Self-Evolving-Neural-Network/backend

# Install dependencies
pip install -r ../requirements.txt

# Run the Flask API
python app.py
```
*The backend will start at `http://localhost:5000`.*

### 2. Frontend Setup

```bash
# Navigate to frontend directory
cd ../frontend

# Install node dependencies
npm install

# Start the development server
npm run dev
```
*The UI will run at `http://localhost:5173`.*

---

## 📖 Usage Guide

### Step 1: Define Your Task
Open the web UI. In the **"Describe your idea"** box, type a description of what you want to achieve.
> *Example: "Create a model to detect pneumonia from chest X-ray images."*

Click **"Predict Category"**. The system will analyze your prompt and select the best task type (e.g., *Image Classification*).

### Step 2: Configure Parameters
-   **Input Shape**: Define the dimensions of your input data.
    -   Image: `128,128,3` (Height, Width, Channels)
    -   NLP: `100` (Sequence Length)
    -   Time Series: `50,1` (Timesteps, Features)
-   **Num Classes**: Number of target categories (or `1` for binary/regression).

### Step 3: Upload Dataset based on Strict Rules
The system enforces strict folder structures to ensure successful training. Prepare your data exactly as shown below:

#### 📸 Image Classification (`.zip`)
A ZIP file containing numbered folders, where the folder name is the class index.
```text
dataset.zip
├── 0/
│   ├── image_01.jpg
│   └── image_02.png
├── 1/
│   ├── image_03.jpg
│   └── ...
```

#### 🖌️ Image Segmentation (`.zip`)
A ZIP file with separate `images` and `masks` folders. Filenames must match.
```text
dataset.zip
├── images/
│   ├── sample_1.jpg
│   └── ...
├── masks/
│   ├── sample_1.jpg  (Grayscale mask)
│   └── ...
```

#### 🎨 Image Generation (`.zip`)
-   **Text-to-Image**: `text.txt` (one prompt per line) and `images/` folder.
-   **Image-to-Image**: `input/` and `output/` folders with matching filenames.

#### 📝 NLP / Time Series (`.csv`)
A single CSV file.
-   **NLP**: Must have columns `X` (text content) and `y` (label).
-   **Time Series**: Must have `y` (target value) and numerical feature columns.

### Step 4: Evolution & Results
Click **"Start Evolution"**.
-   The system will evolve multiple architectures (Depth 1, Depth 2, Depth 3...).
-   The best performing model (highest Validation Accuracy or lowest Loss) is selected.
-   Download your **Trained Model (`.keras`)** and **Tokenizer (`.json`)** directly from the UI.

---

## 📂 Project Structure

```text
Hybrid-Self-Evolving-Neural-Network/
├── backend/
│   ├── app.py                 # Main Flask Application
│   ├── self_evolving_algo.py  # Core Evolutionary Logic
│   ├── dataset_validator.py   # Strict Data Validation Rules
│   ├── model_builder.py       # Keras Model Construction
│   └── uploads/               # Temp storage for user datasets
├── frontend/
│   ├── src/
│   │   ├── components/        # React Components (UI)
│   │   ├── App.jsx            # Main React Logic
│   │   └── ...
│   └── ...
├── model/                     # Pre-trained Controller Models
├── requirements.txt           # Python Dependencies
└── README.md                  # Documentation
```

---

## 🤝 Contribution

Contributions are welcome! Please fork the repository and submit a pull request.
1.  Fork the Project
2.  Create your Feature Branch (`git checkout -b feature/AmazingFeature`)
3.  Commit your Changes (`git commit -m 'Add some AmazingFeature'`)
4.  Push to the Branch (`git push origin feature/AmazingFeature`)
5.  Open a Pull Request

## 📄 License

Distributed under the MIT License. See `LICENSE` for more information.
