import os
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.models import load_model
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# Ścieżki do danych i modelu
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PROCESSED_DIR = os.path.join(BASE_DIR, "data", "processed")
MODEL_DIR = os.path.join(BASE_DIR, "models")
MODEL_PATH = os.path.join(MODEL_DIR, "line_tracking_model.h5")

def load_test_data():
    """
    Wczytuje dane testowe.
    """
    X_test = np.load(os.path.join(PROCESSED_DIR, "X_test.npy"))
    y_test = np.load(os.path.join(PROCESSED_DIR, "y_test.npy"))
    return X_test, y_test

def calculate_metrics(y_true, y_pred):
    """
    Oblicza i wyświetla metryki: MAE, MSE, RMSE, R2.
    """
    mae = mean_absolute_error(y_true, y_pred)
    mse = mean_squared_error(y_true, y_pred)
    rmse = np.sqrt(mse)
    r2 = r2_score(y_true, y_pred)

    print("\n### Metryki ###")
    print(f"MAE (Mean Absolute Error): {mae:.4f}")
    print(f"MSE (Mean Squared Error): {mse:.4f}")
    print(f"RMSE (Root Mean Squared Error): {rmse:.4f}")
    print(f"R² (R-squared): {r2:.4f}")

def plot_results(y_true, y_pred):
    """
    Tworzy wizualizacje: rzeczywiste vs. przewidywane, histogram błędów.
    """
    plt.figure(figsize=(14, 6))

    # Wykres 1: Rzeczywiste vs Przewidywane
    plt.subplot(1, 2, 1)
    plt.scatter(range(len(y_true)), y_true, label='Rzeczywiste', alpha=0.6, color='blue')
    plt.scatter(range(len(y_pred)), y_pred, label='Przewidywane', alpha=0.6, color='orange')
    plt.title("Rzeczywiste vs. Przewidywane kąty skrętu")
    plt.xlabel("Indeks")
    plt.ylabel("Kąt skrętu")
    plt.legend()

    # Wykres 2: Histogram błędów
    errors = y_true - y_pred
    plt.subplot(1, 2, 2)
    plt.hist(errors, bins=20, color='purple', alpha=0.7)
    plt.title("Histogram błędów (y_true - y_pred)")
    plt.xlabel("Błąd")
    plt.ylabel("Liczba przykładów")

    plt.tight_layout()
    plt.show()

def main():
    # Wczytaj model
    model = load_model(MODEL_PATH)
    print(f"Model załadowano z {MODEL_PATH}")

    # Wczytaj dane testowe
    X_test, y_test = load_test_data()

    # Przewiduj kąty skrętu
    y_pred = model.predict(X_test).flatten()

    # Oblicz metryki
    calculate_metrics(y_test, y_pred)

    # Twórz wizualizacje
    plot_results(y_test, y_pred)

if __name__ == "__main__":
    main()
