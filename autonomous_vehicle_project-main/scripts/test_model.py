import numpy as np
from tensorflow.keras.models import load_model
import cv2
import time
import tensorflow as tf
from tensorflow.keras.mixed_precision import set_global_policy

# Włącz Mixed Precision
set_global_policy('mixed_float16')
print("Włączono Mixed Precision dla TensorFlow.")

# Dynamiczna alokacja pamięci GPU
gpus = tf.config.list_physical_devices('GPU')
if gpus:
    try:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
        print("Dynamiczna alokacja pamięci GPU włączona.")
    except RuntimeError as e:
        print(e)

MODEL_PATH = "../models/line_tracking_model.h5"
IMAGE_PATH = "../data/raw/test_drive/output_images/1.png"  # Zmień na rzeczywisty obraz

def preprocess_image(image):
    img = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    img = cv2.resize(img, (112, 112))  # Zmniejsz rozmiar, jeśli potrzeba (np. 112x112)
    img = img / 255.0
    return np.expand_dims(img, axis=0)

# Ładowanie modelu
model = load_model(MODEL_PATH)
print("Model załadowano!")

# Wczytaj obraz
image = cv2.imread(IMAGE_PATH)
if image is None:
    print(f"Błąd: Nie znaleziono obrazu pod ścieżką {IMAGE_PATH}")
    exit(1)

# Mierzenie czasu przetwarzania obrazu
start_time = time.time()
processed_image = preprocess_image(image)
processing_time = time.time() - start_time
print(f"Czas przetwarzania obrazu: {processing_time:.4f} sekund")

# Mierzenie czasu predykcji
start_time = time.time()
predicted_steering_angle = model.predict(processed_image)[0][0]
prediction_time = time.time() - start_time
print(f"Czas predykcji: {prediction_time:.4f} sekund")

# Wyświetlenie wyniku
print(f"Przewidywany kąt skrętu: {predicted_steering_angle}")
