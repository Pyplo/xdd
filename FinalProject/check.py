import cv2
import numpy as np
import os

folder_path = 'C:/Users/Patry/Desktop/FinalProject/Preparing_data/data/dataset/Town04/train/masks'  # Ścieżka do folderu z maskami
folder_path2 = 'C:/Users/Patry/Downloads/archive/val_label'

for file_name in os.listdir(folder_path2):
    if file_name.endswith('.png'):  # Upewnij się, że plik to obraz
        file_path = os.path.join(folder_path2, file_name)
        mask = cv2.imread(file_path, cv2.IMREAD_UNCHANGED)  # Wczytaj obraz w skali szarości
        unique_values = np.unique(mask)
        print(f"Unikalne wartości pikseli w {file_name}: {unique_values}")
import matplotlib.pyplot as plt

for file_name in os.listdir(folder_path2):
    if file_name.endswith('.png'):
        file_path = os.path.join(folder_path2, file_name)
        mask = cv2.imread(file_path, cv2.IMREAD_UNCHANGED)
        plt.imshow(mask, cmap='viridis')  # Użyj mapy kolorów do wizualizacji
        plt.title(file_name)
        plt.show()
