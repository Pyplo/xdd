import carla
import numpy as np
import time
from tensorflow.keras.models import load_model
import cv2

# Ścieżki i parametry
MODEL_PATH = "../models/line_tracking_model.h5"
IMAGE_WIDTH = 112
IMAGE_HEIGHT = 112


def preprocess_image(image):
    """
    Przetwarzanie obrazu: zmiana rozmiaru i normalizacja.
    """
    img = cv2.resize(image, (IMAGE_WIDTH, IMAGE_HEIGHT))
    img = img / 255.0
    return np.expand_dims(img, axis=0)


def control_vehicle(image_data, vehicle):
    """
    Sterowanie pojazdem na podstawie obrazu z kamery.
    """
    # Przetwarzanie obrazu
    image = np.array(image_data.raw_data).reshape((image_data.height, image_data.width, 4))[:, :, :3]
    processed_image = preprocess_image(image)

    # Przewidywanie kąta skrętu
    predicted_steering_angle = model.predict(processed_image)[0][0]

    # Tworzenie komendy sterowania
    control = carla.VehicleControl()
    control.throttle = 0.5  # Stała prędkość
    control.steer = np.clip(predicted_steering_angle / 90, -1.0, 1.0)  # Normalizacja kąta
    vehicle.apply_control(control)

    # Logowanie
    print(f"Przewidywany kąt skrętu: {predicted_steering_angle:.2f}, Steer: {control.steer:.2f}")


def main():
    """
    Główna pętla symulacji.
    """
    # Połączenie z klientem Carla
    client = carla.Client("localhost", 2000)
    client.set_timeout(10.0)
    world = client.get_world()

    # Pobranie blueprintów
    blueprint_library = world.get_blueprint_library()
    vehicle_bp = blueprint_library.filter('vehicle.tesla.model3')[0]

    # Usuwanie istniejących pojazdów i pieszych
    actors = world.get_actors()
    for actor in actors:
        if actor.type_id.startswith("vehicle.") or actor.type_id.startswith("walker."):
            actor.destroy()

    # Sprawdzenie dostępnych punktów spawn
    spawn_points = world.get_map().get_spawn_points()
    if len(spawn_points) == 0:
        print("Brak dostępnych punktów spawn!")
        return

    spawn_point = spawn_points[0]
    vehicle = world.try_spawn_actor(vehicle_bp, spawn_point)

    if vehicle is None:
        print("Nie udało się zespawnować pojazdu. Sprawdź spawn point.")
        return
    else:
        print(f"Zespawnowano pojazd: {vehicle.type_id}")

    # Tworzenie kamery
    camera_bp = blueprint_library.find('sensor.camera.rgb')
    camera_bp.set_attribute("image_size_x", str(IMAGE_WIDTH))
    camera_bp.set_attribute("image_size_y", str(IMAGE_HEIGHT))
    camera_bp.set_attribute("fov", "110")

    camera_transform = carla.Transform(carla.Location(x=2.5, z=0.7))
    camera = world.spawn_actor(camera_bp, camera_transform, attach_to=vehicle)

    # Ładowanie modelu
    global model
    model = load_model(MODEL_PATH)
    print("Model załadowano!")

    # Rejestracja listenera kamery
    camera.listen(lambda image_data: control_vehicle(image_data, vehicle))

    try:
        time.sleep(20)  # Symulacja trwa 20 sekund
    finally:
        print("Zatrzymywanie pojazdu i zamykanie...")
        camera.stop()
        vehicle.destroy()
        camera.destroy()


if __name__ == "__main__":
    main()
