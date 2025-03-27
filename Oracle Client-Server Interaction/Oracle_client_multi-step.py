import socket
import json
import pandas as pd
import os
import sys

def is_valid_csv(file_path):
    try:
        # Check if the file exists
        if not os.path.exists(file_path):
            print(f"Error: File '{file_path}' does not exist")
            return False

        df = pd.read_csv(file_path)

        # Check for required columns
        required_columns = [
            'date', 'max_temp', 'min_temp', 'precipitation', 'cloud_cover',
            'snow_depth', 'pressure', 'sunshine', 'global_radiation'
        ]

        missing_columns = [col for col in required_columns if col not in df.columns]
        if missing_columns:
            print(f"Error: CSV file is missing the following required columns: {', '.join(missing_columns)}")
            return False

        try:
            df['date'] = pd.to_datetime(df['date'])
        except:
            print("Error: 'date' column cannot be converted to datetime format. Please ensure the format is correct (e.g., YYYYMMDD or YYYY-MM-DD)")
            return False

        # Check data quantity
        if len(df) != 30:
            print(f"Error: CSV file has less than 30 days of data; currently only {len(df)} days")
            return False

        return True

    except pd.errors.ParserError:
        print("Error: The file is not in a valid CSV format")
        return False
    except Exception as e:
        print(f"Error: An exception occurred while validating the CSV file: {e}")
        return False

def send_request(file_content):
    server_host = "localhost"
    server_port = 5010

    try:
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as client_socket:
            # Set connection timeout
            client_socket.settimeout(10)

            client_socket.connect((server_host, server_port))
            print(f"Connected to server {server_host}:{server_port}")

            # Build the request data
            request = {"file_content": file_content}
            request_json = json.dumps(request)

            # Send the request
            client_socket.sendall(request_json.encode('utf-8'))
            print("Data sent, waiting for response...")

            # Receive response in chunks
            chunks = []
            while True:
                chunk = client_socket.recv(4096)
                if not chunk:
                    break
                chunks.append(chunk)

                # Check if complete JSON has been received
                try:
                    json.loads(b''.join(chunks).decode('utf-8'))
                    break
                except:
                    pass

            response_data = b''.join(chunks).decode('utf-8')
            return json.loads(response_data)

    except socket.timeout:
        print("Error: Connection to server timed out")
        return {"success": False, "error": "Connection to server timed out"}
    except ConnectionRefusedError:
        print("Error: Connection refused. Please ensure the server is running")
        return {"success": False, "error": "Server connection was refused"}
    except Exception as e:
        print(f"Error: An exception occurred while connecting to the server: {e}")
        return {"success": False, "error": f"Client error: {str(e)}"}

def display_predictions(predictions):
    print("\n===== 5-Day Temperature Prediction =====")
    print("Date          Predicted Temperature (°C)")
    print("--------------------------")
    for pred in predictions:
        print(f"{pred['date']}    {pred['temperature']}°C")
    print("==========================\n")

def client_main():
    print("===== Weather Prediction System - Client =====")
    print("Hello, I’m the Oracle. How can I help you today?")

    while True:
        user_input = input("\nEnter command: ").strip().lower()

        # Check for exit command (exact match)
        if user_input in ["exit"]:
            print("Client exiting")
            break

        # Check for prediction command: must contain "predict" and at least one temperature-related keyword
        if "predict" in user_input and any(keyword in user_input for keyword in ["mean_temperature", "mean temperature", "mean_temp"]):
            file_path = input("Please enter the path to a CSV file that contains historical data for 30 days:").strip()
            if file_path.lower() in ["exit"]:
                print("Client exiting")
                break

            if not is_valid_csv(file_path):
                print("Please provide a valid CSV file or type 'exit' to quit\n")
                continue

            try:
                with open(file_path, 'r', encoding='utf-8') as f:
                    file_content = f.read()

                print("\nSending data to server...")
                response = send_request(file_content)

                if response.get("success", False):
                    display_predictions(response.get("predictions", []))
                    # Exit after a successful 5-day mean temperature prediction
                    sys.exit(0)
                else:
                    print(f"Error: {response.get('error', 'Unknown error')}")
            except Exception as e:
                print(f"Error processing file: {e}")
        else:
            print("I can only perform mean temperature predictions. What else can I help you with?")
            print("If you want to predict mean temperature, please include 'predict' and 'mean temperature' in your input.\n"
                  "If you want to exit, type 'exit'")

if __name__ == "__main__":
    try:
        client_main()
    except KeyboardInterrupt:
        print("\nInterrupt signal received, client exiting")
    except Exception as e:
        print(f"Client encountered an error: {e}")
