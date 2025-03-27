import socket
import json
import pandas as pd
import numpy as np
import joblib
import io
import threading
from tensorflow.keras.models import load_model

# load model and scalers
def load_model_and_scalers():
    global model, feature_scaler, target_scaler

    try:
        print("Loading model and scalers...")
        model = load_model('multi-step_model.h5')
        feature_scaler = joblib.load('feature_scaler_multi-step.pkl')
        target_scaler = joblib.load('target_scaler_multi-step.pkl')
        print("Model and scalers loaded successfully")
        return True
    except Exception as e:
        print(f"Model loading failed: {e}")
        return False

def preprocess_data(df):
    # Ensure date format conversion
    df['date'] = pd.to_datetime(df['date'])

    # Extract time features
    df['year'] = df['date'].dt.year
    df['month'] = df['date'].dt.month
    df['day'] = df['date'].dt.day

    # Ensure the input feature order matches the training order
    feature_columns = [
        'max_temp', 'min_temp', 'precipitation', 'cloud_cover',
        'snow_depth', 'pressure', 'sunshine', 'global_radiation',
        'year', 'month', 'day'
    ]

    # Check for missing columns
    missing_cols = [col for col in feature_columns if col not in df.columns]
    if missing_cols:
        raise ValueError(f"CSV file is missing the following required columns: {missing_cols}")

    # Handle missing values (linear interpolation)
    input_data = df[feature_columns].interpolate(method='linear')

    # Handle outliers (Winsorization), consistent with training process
    Q1 = input_data.quantile(0.25)
    Q3 = input_data.quantile(0.75)
    IQR = Q3 - Q1
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR
    input_data = input_data.clip(lower=lower_bound, upper=upper_bound, axis=1)

    return input_data

def process_prediction(file_content):
    try:
        df = pd.read_csv(io.StringIO(file_content))
        print(f"CSV data parsed successfully, total {len(df)} rows")

        # Ensure the data contains 30 days
        if len(df) != 30:
            return json.dumps({
                "success": False,
                "error": f"Not enough days in input data; only {len(df)} days provided"
            })

        # Save the original dates for generating prediction dates later
        if 'date' in df.columns:
            # Try different date format conversions
            try:
                # First, try standard format %Y%m%d
                df['date'] = pd.to_datetime(df['date'], format='%Y%m%d')
            except:
                try:
                    # Then try ISO format %Y-%m-%d
                    df['date'] = pd.to_datetime(df['date'])
                except Exception as e:
                    return json.dumps({
                        "success": False,
                        "error": f"Date conversion failed: {e}. Please ensure the date format in the CSV is correct."
                    })

            # Sort data by date
            df = df.sort_values('date')

            # Get the last observed date
            last_date = df['date'].iloc[-1]
            print(f"Last observation date: {last_date}")
        else:
            return json.dumps({
                "success": False,
                "error": "CSV file is missing the 'date' column"
            })

        # Preprocess the data
        input_data = preprocess_data(df)

        input_data = input_data.iloc[-30:].reset_index(drop=True)

        # Normalize the input features
        X_scaled = feature_scaler.transform(input_data)

        # Reshape into LSTM input format [samples, timesteps, features]
        X_input = np.reshape(X_scaled, (1, X_scaled.shape[0], X_scaled.shape[1]))

        y_pred_scaled = model.predict(X_input)

        # Inverse normalization of prediction results
        y_pred_scaled_reshaped = y_pred_scaled.reshape(-1, 1)
        y_pred = target_scaler.inverse_transform(y_pred_scaled_reshaped).flatten()

        # Generate prediction dates starting from the last observation date
        results = []
        for i, temp in enumerate(y_pred):
            # Calculate each future prediction date
            future_date = last_date + pd.Timedelta(days=i + 1)

            date_str = future_date.strftime('%Y-%m-%d')
            print(
                f"Prediction {i + 1}: Original date={last_date}, Computed date={future_date}, Formatted date={date_str}, Temperature={round(float(temp), 2)}")

            results.append({
                "date": date_str,
                "day": i + 1,
                "temperature": round(float(temp), 2)
            })

        return json.dumps({
            "success": True,
            "predictions": results
        })

    except Exception as e:
        print(f"Error during prediction processing: {e}")
        import traceback
        traceback.print_exc()  # Print full error stack
        return json.dumps({
            "success": False,
            "error": str(e)
        })

def handle_client(client_socket, client_address):
    print(f"New connection from: {client_address}")
    try:
        client_socket.settimeout(30)

        chunks = []
        while True:
            chunk = client_socket.recv(4096)
            if not chunk:
                break
            chunks.append(chunk)

            # Check if the JSON end marker is present
            if b'"file_content":' in b''.join(chunks) and chunk.endswith(b'}'):
                break

        data_received = b''.join(chunks).decode('utf-8')

        # Parse JSON request
        try:
            request = json.loads(data_received)
            file_content = request.get('file_content', '')

            if not file_content:
                response = json.dumps({
                    "success": False,
                    "error": "No CSV file content received"
                })
            else:
                response = process_prediction(file_content)

        except json.JSONDecodeError:
            print("Received invalid JSON data")
            response = json.dumps({
                "success": False,
                "error": "Invalid JSON format"
            })

        # Send response
        client_socket.sendall(response.encode('utf-8'))
        print(f"Response sent to client {client_address}")

    except socket.timeout:
        print(f"Client {client_address} connection timed out")
        client_socket.sendall(json.dumps({
            "success": False,
            "error": "Connection timed out"
        }).encode('utf-8'))

    except Exception as e:
        print(f"Error handling client {client_address}: {e}")
        try:
            client_socket.sendall(json.dumps({
                "success": False,
                "error": f"Server error: {str(e)}"
            }).encode('utf-8'))
        except:
            pass

    finally:
        client_socket.close()
        print(f"Closed connection with client {client_address}")

def start_server(host='localhost', port=5010):
    # First, load the model and scalers
    if not load_model_and_scalers():
        print("Unable to start server, model loading failed")
        return

    server_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    server_socket.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)

    try:
        server_socket.bind((host, port))
        server_socket.listen(5)
        print(f"Server started, listening on {host}:{port}")

        while True:
            client_socket, client_address = server_socket.accept()
            client_thread = threading.Thread(
                target=handle_client,
                args=(client_socket, client_address)
            )
            client_thread.daemon = True
            client_thread.start()

    except KeyboardInterrupt:
        print("Interrupt signal received, shutting down server...")
    except Exception as e:
        print(f"Server error: {e}")
    finally:
        server_socket.close()
        print("Server closed")

if __name__ == '__main__':
    start_server()
