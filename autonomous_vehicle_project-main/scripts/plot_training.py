import os
import numpy as np
import matplotlib.pyplot as plt

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODEL_DIR = os.path.join(BASE_DIR, "models")

def plot_training_history(history_path):
    """
    Wizualizuje historię uczenia.
    """
    if not os.path.exists(history_path):
        raise FileNotFoundError(f"Nie znaleziono historii uczenia: {history_path}")

    history = np.load(history_path, allow_pickle=True).item()

    # Wykres strat
    plt.figure(figsize=(10, 5))
    plt.subplot(1, 2, 1)
    plt.plot(history['loss'], label='Loss (train)', color='blue')
    plt.plot(history['val_loss'], label='Loss (val)', color='orange')
    plt.title("Krzywa strat (Loss)")
    plt.xlabel("Epoki")
    plt.ylabel("Loss")
    plt.legend()

    # Wykres dokładności (MAE)
    plt.subplot(1, 2, 2)
    plt.plot(history['mae'], label='MAE (train)', color='blue')
    plt.plot(history['val_mae'], label='MAE (val)', color='orange')
    plt.title("Krzywa MAE")
    plt.xlabel("Epoki")
    plt.ylabel("MAE")
    plt.legend()

    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    history_path = os.path.join(MODEL_DIR, "training_history.npy")
    plot_training_history(history_path)
