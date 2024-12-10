import os
import numpy as np
import matplotlib.pyplot as plt

# Ścieżki do przetworzonych danych
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PROCESSED_DIR = os.path.join(BASE_DIR, "data", "processed")
X_PATH = os.path.join(PROCESSED_DIR, "X_all_data.npy")
Y_PATH = os.path.join(PROCESSED_DIR, "y_all_data.npy")

def inspect_data():
    # Wczytanie danych
    X = np.load(X_PATH)
    y = np.load(Y_PATH)

    print(f"Załadowano dane:")
    print(f"X shape: {X.shape}, y shape: {y.shape}")

    # Wyświetlenie kilku przykładów
    for i in range(5):  # Wyświetl 5 pierwszych obrazów
        plt.figure(figsize=(4, 4))
        plt.imshow(X[i])
        plt.title(f"Kąt skrętu: {y[i]:.2f}")
        plt.axis('off')
        plt.show()

if __name__ == "__main__":
    inspect_data()
