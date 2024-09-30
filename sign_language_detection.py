import tkinter as tk
from tkinter import filedialog
import cv2
import tensorflow as tf
import numpy as np
from PIL import Image, ImageTk
from datetime import datetime

# Preprocess the image/frame for prediction
def preprocess_frame(frame, img_size=(64, 64)):
    gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    resized_frame = cv2.resize(gray_frame, img_size)
    normalized_frame = resized_frame / 255.0
    return np.expand_dims(np.expand_dims(normalized_frame, axis=0), axis=-1)

def real_time_detection(model):
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Error: Could not open webcam.")
        return
    
    while True:
        ret, frame = cap.read()
        if not ret:
            print("Error: Failed to capture image")
            break

        # Preprocess the frame
        processed_frame = preprocess_frame(frame)
        
        # Predict the sign language gesture
        prediction = model.predict(processed_frame)
        sign = np.argmax(prediction)
        
        # Display the prediction on the frame
        cv2.putText(frame, f"Sign: {sign}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)
        
        # Display the resulting frame
        cv2.imshow('Real-Time Sign Language Detection', frame)
        
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

def upload_image():
    file_path = filedialog.askopenfilename()
    img = preprocess_frame(cv2.imread(file_path))
    prediction = model.predict(np.expand_dims(img, axis=0))
    sign = np.argmax(prediction)
    result_label.config(text=f"Predicted Sign: {sign}")

def start_detection():
    if check_time():
        real_time_detection(model)
    else:
        print("Application is only operational between 6 PM and 10 PM.")

def check_time():
    current_time = datetime.now().time()
    start_time = datetime.strptime("18:00:00", "%H:%M:%S").time()
    end_time = datetime.strptime("22:00:00", "%H:%M:%S").time()
    return start_time <= current_time <= end_time

# Load the model
model = tf.keras.models.load_model('sign_language_model.h5')

# Create the GUI
root = tk.Tk()
root.title("Sign Language Detection")

upload_button = tk.Button(root, text="Upload Image", command=upload_image)
upload_button.pack()

detect_button = tk.Button(root, text="Start Real-Time Detection", command=start_detection)
detect_button.pack()

result_label = tk.Label(root, text="")
result_label.pack()

root.mainloop()
