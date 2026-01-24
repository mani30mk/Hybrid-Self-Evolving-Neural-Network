import requests
import os

BASE_URL = "http://127.0.0.1:5000/api"
DATASET_PATH = "dataset/test_data.csv"

def test_category_prediction():
    print("\n[1] Testing Category Prediction...")
    try:
        payload = {"prompt": "Create a simple neural network for binary classification of tabular data."}
        response = requests.post(f"{BASE_URL}/predict-category", json=payload)
        response.raise_for_status()
        data = response.json()
        print(f"✅ Success! Category: {data.get('category')}")
        return data.get('category')
    except Exception as e:
        print(f"❌ Failed: {e}")
        try:
            print(f"Server response: {response.text}")
        except:
            pass
        return None

def test_architecture_prediction(category):
    print("\n[2] Testing Architecture Prediction & Training...")
    if not os.path.exists(DATASET_PATH):
        print(f"❌ Dataset not found at {DATASET_PATH}")
        return

    try:
        # Simulate Form Data
        files = {
            'dataset': ('test_data.csv', open(DATASET_PATH, 'rb'), 'text/csv')
        }
        data = {
            'prompt': "Create a simple neural network for binary classification of tabular data.",
            'category': "Time Series / Audio", # Force correct category for tabular
            'input_shape': "2",
            'num_classes': "2"
        }

        print("Sending request... (this may take a moment)")
        response = requests.post(f"{BASE_URL}/predict-architecture", files=files, data=data)

        if response.status_code == 200:
             result = response.json()
             print("✅ Success! Model Trained.")
             print(f"   - Layers: {result.get('num_layers')}")
             print(f"   - Params: {result.get('num_parameters')}")
             print(f"   - Accuracy: {result.get('final_val_accuracy'):.4f}")
        else:
             print(f"❌ Failed with Status {response.status_code}")
             print(f"   Error: {response.text}")

    except Exception as e:
        print(f"❌ Exception: {e}")

if __name__ == "__main__":
    cat = test_category_prediction()
    if cat:
        test_architecture_prediction(cat)
