import os
import numpy as np
import pandas as pd
import cv2
from sklearn.model_selection import train_test_split

# Ścieżki bazowe
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RAW_DIR = os.path.join(BASE_DIR, "data", "raw")
PROCESSED_DIR = os.path.join(BASE_DIR, "data", "processed")

# Parametry obrazu
IMAGE_SIZE = (112, 112)  # Zmniejszony rozmiar obrazów

def preprocess_image(img):
    """
    Przekształca obraz: zmiana rozmiaru i normalizacja.
    """
    img = cv2.resize(img, IMAGE_SIZE)  # Zmiana rozmiaru
    img = img / 255.0  # Normalizacja
    return img

def process_scenario(scenario_name):
    """
    Przetwarza dane z jednego scenariusza.
    """
    scenario_dir = os.path.join(RAW_DIR, scenario_name)
    images_dir = os.path.join(scenario_dir, "output_images")
    csv_path = os.path.join(scenario_dir, "vehicle_data.csv")

    if not os.path.exists(images_dir) or not os.path.exists(csv_path):
        raise FileNotFoundError(f"Dane scenariusza '{scenario_name}' są niekompletne!")

    # Wczytanie danych z CSV
    data = pd.read_csv(csv_path)
    print(f"Przetwarzanie scenariusza: {scenario_name} ({len(data)} ramek)")

    images, angles = [], []
    for _, row in data.iterrows():
        frame = int(row['frame'])
        img_path = os.path.join(images_dir, f"{frame}.png")

        if not os.path.exists(img_path):
            print(f"Brak obrazu: {img_path}, pomijanie...")
            continue

        img = cv2.imread(img_path)
        if img is None:
            print(f"Nie można wczytać obrazu: {img_path}, pomijanie...")
            continue

        images.append(preprocess_image(img))
        angles.append(row['yaw'])

    return np.array(images), np.array(angles)

def split_and_save_data(images, angles):
    """
    Dzieli dane na train/test i zapisuje je w plikach .npy.
    """
    X_train, X_test, y_train, y_test = train_test_split(images, angles, test_size=0.2, random_state=42)

    np.save(os.path.join(PROCESSED_DIR, "X_train.npy"), X_train)
    np.save(os.path.join(PROCESSED_DIR, "y_train.npy"), y_train)
    np.save(os.path.join(PROCESSED_DIR, "X_test.npy"), X_test)
    np.save(os.path.join(PROCESSED_DIR, "y_test.npy"), y_test)

    print(f"Zapisano dane: Train ({len(X_train)}), Test ({len(X_test)})")

def main():
    scenarios = [d for d in os.listdir(RAW_DIR) if os.path.isdir(os.path.join(RAW_DIR, d))]
    if not scenarios:
        raise RuntimeError("Brak scenariuszy w folderze raw!")

    all_images, all_angles = [], []
    for scenario in scenarios:
        images, angles = process_scenario(scenario)
        all_images.extend(images)
        all_angles.extend(angles)

    split_and_save_data(np.array(all_images), np.array(all_angles))

if __name__ == "__main__":
    if not os.path.exists(PROCESSED_DIR):
        os.makedirs(PROCESSED_DIR)
    main()
