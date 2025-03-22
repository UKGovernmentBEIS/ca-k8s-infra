from flask import Flask, request, jsonify
import os

app = Flask(__name__)

# Archivo para almacenar los logs
LOG_FILE = "logs.txt"

@app.route('/')
def hello():
    return "¡Hola desde el servicio Hello!"

@app.route('/write_log', methods=['POST'])
def write_log():
    data = request.json
    if not data or 'message' not in data:
        return jsonify({"error": "Se requiere un mensaje"}), 400
    
    # Escribir la línea de log en el archivo
    with open(LOG_FILE, 'a') as f:
        f.write(f"{data['message']}\n")
    
    return jsonify({"status": "success", "message": "Log escrito correctamente"})

@app.route('/read_logs', methods=['GET'])
def read_logs():
    # Verificar si el archivo existe
    if not os.path.exists(LOG_FILE):
        return jsonify({"logs": []})
    
    # Leer todas las líneas del archivo
    with open(LOG_FILE, 'r') as f:
        logs = f.readlines()
    
    # Eliminar caracteres de nueva línea y espacios en blanco
    logs = [log.strip() for log in logs]
    
    return jsonify({"logs": logs})

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=8081)
