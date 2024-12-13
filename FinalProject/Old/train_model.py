import cv2
import numpy as np
import tensorflow as tf
import json
import os


class LaneDetector:
    def __init__(self, input_shape=(720, 1280, 3)):
        self.input_shape = input_shape
        self.model = None
        self.build_model()

    def build_model(self):
        """
        Budowa sieci neuronowej dla detekcji linii
        """
        model = tf.keras.Sequential([
            tf.keras.layers.Input(shape=self.input_shape),
            tf.keras.layers.Conv2D(64, (3, 3), activation='relu', padding='same'),
            tf.keras.layers.MaxPooling2D((2, 2)),
            tf.keras.layers.Conv2D(128, (3, 3), activation='relu', padding='same'),
            tf.keras.layers.MaxPooling2D((2, 2)),
            tf.keras.layers.Conv2D(256, (3, 3), activation='relu', padding='same'),
            tf.keras.layers.Flatten(),
            tf.keras.layers.Dense(512, activation='relu'),
            tf.keras.layers.Dense(256, activation='linear')
            # Przewidywanie współrzędnych linii
        ])

        model.compile(optimizer='adam', loss='mse')
        self.model = model

    def preprocess_image(self, image):
        """
        Preprocessing obrazu
        """
        # Konwersja do skali szarości
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

        # Rozmycie Gaussa
        blurred = cv2.GaussianBlur(gray, (5, 5), 0)

        # Detekcja krawędzi Canny
        edges = cv2.Canny(blurred, 50, 150)

        return edges

    def detect_lanes(self, image):
        """
        Główna metoda detekcji linii
        """
        # Preprocessing
        processed = self.preprocess_image(image)

        # Transformata Hougha do wykrycia linii
        lines = cv2.HoughLinesP(
            processed,
            rho=1,
            theta=np.pi / 180,
            threshold=50,
            minLineLength=50,
            maxLineGap=10
        )

        return lines

    def draw_lanes(self, image, lines):
        """
        Rysowanie wykrytych linii
        """
        lane_image = image.copy()

        if lines is not None:
            for line in lines:
                x1, y1, x2, y2 = line[0]
                cv2.line(lane_image, (x1, y1), (x2, y2), (0, 255, 0), 5)

        return lane_image


def process_dataset(json_path):
    """
    Przetwarzanie zbioru danych z pliku JSON
    """
    with open(json_path, 'r') as f:
        dataset = [json.loads(line) for line in f]

    images = []
    lane_coordinates = []

    for data in dataset:
        # Wczytanie obrazu
        img_path = data['raw_file']
        image = cv2.imread(img_path)

        # Ekstrakcja współrzędnych linii
        lanes = data['lanes']
        h_samples = data['h_samples']

        # Przetwarzanie współrzędnych
        processed_lanes = []
        for lane in lanes:
            lane_coords = [coord for coord in lane if coord != -2]
            if lane_coords:
                processed_lanes.append(lane_coords)

        images.append(image)
        lane_coordinates.append(processed_lanes)

    return images, lane_coordinates


def main():
    # Ścieżka do pliku JSON z danymi
    json_path = '/data/dataset/Town03/small_train_labels_10%.json'

    # Wczytanie danych
    images, lane_coords = process_dataset(json_path)

    # Inicjalizacja detektora
    lane_detector = LaneDetector()

    # Przetwarzanie obrazów
    for img, lanes in zip(images, lane_coords):
        # Detekcja linii
        detected_lines = lane_detector.detect_lanes(img)

        # Rysowanie linii
        result_img = lane_detector.draw_lanes(img, detected_lines)

        # Wyświetlenie wyniku
        cv2.imshow('Lane Detection', result_img)
        cv2.waitKey(0)

    cv2.destroyAllWindows()


if __name__ == '__main__':
    main()