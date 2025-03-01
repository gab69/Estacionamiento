from flask import Flask, render_template
from flask_socketio import SocketIO, emit
from ultralytics import YOLO
import cv2
import base64
import threading
import time


app = Flask(__name__)
socketio = SocketIO(app, async_mode='threading')

# Cargar el modelo YOLO
model = YOLO("Estacionamiento.pt") 

# Iniciar la cámara
cap = cv2.VideoCapture(0)


if not cap.isOpened():
    print("Error: No se pudo abrir la cámara.")
    exit()


is_camera_running = False

# Variables para almacenar la última etiqueta detectada y los contadores
last_label = None
free_count = 0
occupied_count = 0
total_spaces = 10  # Define el número total de espacios en el estacionamiento

# Función para capturar y procesar frames
def capture_frames():
    global is_camera_running, last_label, free_count, occupied_count
    while True:
        if is_camera_running:
            # Capturar un frame de la cámara
            ret, frame = cap.read()
            if not ret:
                print("Error: No se pudo capturar el frame.")
                break

            # Redimensionar el frame para mejorar el rendimiento
            resized_frame = cv2.resize(frame, (640, 480))

            # Realizar predicción con el modelo YOLO
            results = model(resized_frame, conf=0.5, iou=0.45)

            # Obtener el frame con las anotaciones
            annotated_frame = results[0].plot()

            # Comprimir el frame en formato JPEG
            _, buffer = cv2.imencode('.jpg', annotated_frame, [int(cv2.IMWRITE_JPEG_QUALITY), 70])
            frame_base64 = base64.b64encode(buffer).decode('utf-8')

            # Obtener las etiquetas detectadas
            current_labels = results[0].boxes.cls.tolist()  # Obtener las clases detectadas

            # Reiniciar los contadores
            free_count = 0
            occupied_count = 0

            # Contar las etiquetas "libre" y "ocupado"
            for label in current_labels:
                label_name = model.names[int(label)]  
                if label_name == "libre":
                    free_count += 1
                elif label_name == "ocupado":
                    occupied_count += 1

            # Calcular los espacios disponibles
            available_spaces = total_spaces - occupied_count

            # Enviar el frame y los contadores al cliente a través de WebSocket
            socketio.emit('video_frame', {
                'frame': frame_base64,
                'free_count': free_count,
                'occupied_count': occupied_count,
                'available_spaces': available_spaces
            })

           
            if current_labels:
                current_label = model.names[int(current_labels[0])]  
            else:
                current_label = None

            if current_label != last_label and current_label == "dormido":
                last_label = current_label
                socketio.emit('label_changed', current_label)  
            else:
                last_label = current_label

  
        time.sleep(0.1)

# Ruta principal de la aplicación
@app.route('/')
def index():
    return render_template('index.html')

# Manejar el evento de inicio de la cámara
@socketio.on('start_camera')
def handle_start_camera():
    global is_camera_running
    is_camera_running = True
    print("Cámara iniciada.")

# Manejar el evento de detención de la cámara
@socketio.on('stop_camera')
def handle_stop_camera():
    global is_camera_running
    is_camera_running = False
    print("Cámara detenida.")

# Iniciar la captura de frames en un hilo separado
thread = threading.Thread(target=capture_frames)
thread.daemon = True
thread.start()

# Ejecutar la aplicación
if __name__ == '__main__':
    socketio.run(app, host='0.0.0.0', port=5000)