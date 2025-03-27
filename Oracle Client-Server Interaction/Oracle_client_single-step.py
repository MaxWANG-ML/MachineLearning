import socket
import json
import numpy as np

# Server IP and port (must match the server configuration)
SERVER_IP = "127.0.0.1"
SERVER_PORT = 5005

def get_user_input():
    features = ['max_temp', 'min_temp', 'precipitation', 'pressure',
                'cloud_cover', 'snow_depth', 'sunshine', 'global_radiation',
                'year', 'month', 'day']
    user_data = []
    print("Please enter the weather data for the past 5 days, one day per line, with values separated by spaces.")
    print("Format: max_temp min_temp precipitation cloud_cover snow_depth pressure sunshine global_radiation year month day")
    for i in range(5):
        while True:
            try:
                values = input(f"Data for day {i + 1}: ").strip().split()
                values = [float(v) for v in values]
                if len(values) != len(features):
                    print("Incorrect number of values entered. Please try again.")
                    continue
                user_data.append(values)
                break
            except ValueError:
                print("Invalid input format. Please enter numeric values.")
    return user_data

def send_data_to_server(data):
    try:
        sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        sock.connect((SERVER_IP, SERVER_PORT))
        message = json.dumps(data)
        sock.send(message.encode("utf-8"))
        response = sock.recv(4096).decode("utf-8")
        result = json.loads(response)
        sock.close()
        return result
    except Exception as e:
        print("Connection failed:", e)
        return None

def main():
    print("===== Weather Prediction System - Client =====")
    print("Hello, I'm the Oracle. How can I help you today?")
    while True:
        user_command = input("\nEnter command: ").strip().lower()
        if user_command == "exit":
            print("Client exiting.")
            break
        # Check if the input command contains 'predict' and any temperature-related keywords.
        if "predict" in user_command and any(kw in user_command for kw in ["mean_temperature", "mean temperature", "mean_temp"]):
            # When a prediction command is detected, prompt the user to enter weather data.
            data = get_user_input()
            result = send_data_to_server(data)
            if result:
                if "predicted_temp" in result:
                    print(f"\nPredicted temperature for the next day: {result['predicted_temp']:.2f}°C")
                else:
                    print("Server error:", result.get("error", "Unknown error"))
            else:
                print("No response from server.")
            break  
        else:
            print("I can only perform mean temperature predictions. What else can I help you with?")
            print("If you want to predict mean temperature, please include 'predict' and 'mean_temperature' in your input.")
            print("If you want to exit, type 'exit'.")

if __name__ == '__main__':
    main()
