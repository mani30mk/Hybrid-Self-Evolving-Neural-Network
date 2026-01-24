import requests
import os

# Ensure complex dataset exists
if not os.path.exists("dataset/complex_data.csv"):
    print("Error: dataset/complex_data.csv not found.")
    exit(1)

url = "http://localhost:5000/api/predict-architecture"

payload = {
    "prompt": "Create a classifier",
    "category": "Time Series / Audio",
    "input_shape": "10",
    "num_classes": "3"
}

files = {
    "dataset": open("dataset/complex_data.csv", "rb")
}

try:
    print(f"Sending request to {url}...")
    response = requests.post(url, data=payload, files=files)

    print(f"Status Code: {response.status_code}")
    print("Response Body:")
    print(response.text)

    if response.status_code == 200:
        print("\nSUCCESS: Backend processed the request without crashing.")
    else:
        print("\nFAILURE: Backend returned an error.")

except Exception as e:
    print(f"\nCRITICAL FAILURE: Could not connect to backend. {e}")
