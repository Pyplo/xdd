import os
import time
import csv
import carla
import cv2
import numpy as np

class DataCollector:
    def __init__(self, scenario_name):
        self.base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        self.data_dir = os.path.join(self.base_dir, "data", "raw", scenario_name)
        self.images_dir = os.path.join(self.data_dir, "output_images")
        self.csv_path = os.path.join(self.data_dir, "vehicle_data.csv")

        # Tworzenie folderów, jeśli nie istnieją
        os.makedirs(self.images_dir, exist_ok=True)

        # Otwórz plik CSV do zapisu danych
        self.csv_file = open(self.csv_path, 'w', newline='')
        self.csv_writer = csv.writer(self.csv_file)
        self.csv_writer.writerow(['frame', 'x', 'y', 'z', 'pitch', 'yaw', 'roll', 'speed'])

        self.frame = 0
        self.latest_image = None  # Do przechowywania ostatniego obrazu z kamery

    def camera_callback(self, image):
        """Callback do obsługi obrazu z kamery"""
        array = np.frombuffer(image.raw_data, dtype=np.uint8)
        array = array.reshape((image.height, image.width, 4))  # Obraz RGBA
        self.latest_image = array[:, :, :3]  # Zapisujemy tylko RGB

    def save_data(self, vehicle):
        # Pobranie danych o położeniu pojazdu
        transform = vehicle.get_transform()
        velocity = vehicle.get_velocity()

        x, y, z = transform.location.x, transform.location.y, transform.location.z
        pitch, yaw, roll = transform.rotation.pitch, transform.rotation.yaw, transform.rotation.roll
        speed = (velocity.x ** 2 + velocity.y ** 2 + velocity.z ** 2) ** 0.5

        # Zapis danych do CSV
        self.csv_writer.writerow([self.frame, x, y, z, pitch, yaw, roll, speed])

        # Zapis obrazu, jeśli jest dostępny
        if self.latest_image is not None:
            img_path = os.path.join(self.images_dir, f"{self.frame}.png")
            cv2.imwrite(img_path, self.latest_image)

        self.frame += 1

    def close(self):
        self.csv_file.close()
        print(f"Dane zapisano w: {self.data_dir}")


def main():
    client = carla.Client('localhost', 2000)
    client.set_timeout(10.0)

    world = client.get_world()
    blueprint_library = world.get_blueprint_library()

    # Debug: Sprawdzenie aktorów w świecie CARLA
    actors = world.get_actors()
    vehicles = actors.filter('vehicle.*')
    print(f"Liczba pojazdów w świecie: {len(vehicles)}")
    for v in vehicles:
        print(f"Pojazd ID: {v.id}, Typ: {v.type_id}")

    if not vehicles:
        raise RuntimeError("Nie znaleziono żadnych pojazdów w świecie! Upewnij się, że uruchomiłeś manual_control.py.")

    # Wybór pojazdu: ręczny lub automatyczny
    use_manual_selection = input("Czy chcesz ręcznie wybrać pojazd? (tak/nie): ").lower() == "tak"
    if use_manual_selection:
        vehicle_id = int(input("Podaj ID pojazdu: "))
        vehicle = world.get_actor(vehicle_id)
        if not vehicle:
            raise RuntimeError(f"Nie znaleziono pojazdu o ID {vehicle_id}.")
    else:
        # Domyślnie wybieramy ostatni pojazd
        vehicle = vehicles[-1]

    print(f"Wybrano pojazd: {vehicle.type_id}")

    # Dodanie kamery
    camera_bp = blueprint_library.find('sensor.camera.rgb')
    camera_transform = carla.Transform(carla.Location(x=2.5, z=0.7))
    camera = world.spawn_actor(camera_bp, camera_transform, attach_to=vehicle)

    # Inicjalizacja DataCollector
    scenario_name = input("Podaj nazwę scenariusza (np. straight_drive, sharp_turn): ")
    data_collector = DataCollector(scenario_name)

    # Podłącz callback do kamery
    camera.listen(data_collector.camera_callback)

    try:
        print("Zbieranie danych... Naciśnij CTRL+C, aby przerwać.")
        while True:
            data_collector.save_data(vehicle)
            time.sleep(0.1)  # Zbieraj dane co 0.1 sekundy
    except KeyboardInterrupt:
        print("\nZbieranie danych zakończone.")
    finally:
        data_collector.close()
        camera.stop()
        camera.destroy()


if __name__ == "__main__":
    main()
