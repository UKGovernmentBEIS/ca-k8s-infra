from flask import Flask, request, jsonify
import os

app = Flask(__name__)

# File to store logs
LOG_FILE = "logs.txt"

@app.route('/')
def hello():
    return "Hello from the Logging service!"

@app.route('/write_log', methods=['POST'])
def write_log():
    data = request.json
    if not data or 'message' not in data:
        return jsonify({"error": "A message is required"}), 400
    
    # Write the log line to the file
    with open(LOG_FILE, 'a') as f:
        f.write(f"{data['message']}\n")
    
    return jsonify({"status": "success", "message": "Log written successfully"})

@app.route('/read_logs', methods=['GET'])
def read_logs():
    # Check if the file exists
    if not os.path.exists(LOG_FILE):
        return jsonify({"logs": []})
    
    # Read all lines from the file
    with open(LOG_FILE, 'r') as f:
        logs = f.readlines()
    
    # Remove newline characters and whitespace
    logs = [log.strip() for log in logs]
    
    return jsonify({"logs": logs})

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=8081)
