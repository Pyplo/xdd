from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense
import numpy as np
import os

# Ścieżki do danych
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PROCESSED_DIR = os.path.join(BASE_DIR, "data", "processed")
MODEL_DIR = os.path.join(BASE_DIR, "models")

# Parametry obrazu
IMAGE_SIZE = (112, 112, 3)

# Ładowanie danych
X_train = np.load(os.path.join(PROCESSED_DIR, "X_train.npy"))
y_train = np.load(os.path.join(PROCESSED_DIR, "y_train.npy"))
X_test = np.load(os.path.join(PROCESSED_DIR, "X_test.npy"))
y_test = np.load(os.path.join(PROCESSED_DIR, "y_test.npy"))

# Definicja modelu
model = Sequential([
    Conv2D(32, (3, 3), activation='relu', input_shape=IMAGE_SIZE),
    MaxPooling2D((2, 2)),
    Conv2D(64, (3, 3), activation='relu'),
    MaxPooling2D((2, 2)),
    Flatten(),
    Dense(128, activation='relu'),
    Dense(1)  # Wyjście - przewidywany kąt skrętu
])

model.compile(optimizer='adam', loss='mean_squared_error', metrics=['mae'])

# Trening modelu
history = model.fit(X_train, y_train, validation_data=(X_test, y_test), epochs=30, batch_size=32)

# Zapis modelu i historii treningu
if not os.path.exists(MODEL_DIR):
    os.makedirs(MODEL_DIR)

model.save(os.path.join(MODEL_DIR, "line_tracking_model.h5"))
np.save(os.path.join(MODEL_DIR, "training_history.npy"), history.history)

print("Model zapisano!")
