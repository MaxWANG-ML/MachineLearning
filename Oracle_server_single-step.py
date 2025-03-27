import socket
import json
import numpy as np
import joblib
from tensorflow.keras.models import load_model
import threading


def load_model_and_scalers():
    global model, feature_scaler, target_scaler
    try:
        feature_scaler = joblib.load("feature_scaler_single-step.pkl")  # Normalize input features
        target_scaler = joblib.load("target_scaler_single-step.pkl")    # Inverse normalization for output
        model = load_model("single-step_model.h5")
        print("Scalers and LSTM model loaded successfully")
    except Exception as e:
        print("Failed to load model or scalers:", e)
        exit(1)

def predict_next_day(last_5_days):
    arr = np.array(last_5_days)
    arr_scaled = feature_scaler.transform(arr)
    arr_scaled = arr_scaled.reshape(1, 5, -1)  # Reshape to (1, 5, num_features)
    pred_scaled = model.predict(arr_scaled)
    pred = target_scaler.inverse_transform(pred_scaled.reshape(-1, 1))[0, 0]
    return float(pred)

def handle_client(client_socket, client_address):
    print("Client connected:", client_address)
    try:
        data = client_socket.recv(4096).decode("utf-8")
        # Expecting a JSON formatted list of 5 days of data
        weather_data = json.loads(data)
        if isinstance(weather_data, list) and len(weather_data) == 5:
            prediction = predict_next_day(weather_data)
            response = json.dumps({"predicted_temp": prediction})
        else:
            response = json.dumps({"error": "Invalid data format. Expected a list of 5 days of weather data."})
        client_socket.send(response.encode("utf-8"))
    except Exception as e:
        error_response = json.dumps({"error": str(e)})
        client_socket.send(error_response.encode("utf-8"))
    finally:
        client_socket.close()

def start_server(host="127.0.0.1", port=5005):
    load_model_and_scalers()
    server_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    server_socket.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    server_socket.bind((host, port))
    server_socket.listen(5)
    print(f"Server started, listening on {host}:{port}")

    while True:
        client_socket, client_address = server_socket.accept()
        thread = threading.Thread(target=handle_client, args=(client_socket, client_address))
        thread.daemon = True
        thread.start()

if __name__ == '__main__':
    start_server()
