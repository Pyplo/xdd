import glob
import os
import sys
import random
import carla
import pygame
import numpy as np
import queue
from collections import deque
import cv2

try:
    sys.path.append(glob.glob('../../carla/dist/carla-*%d.%d-%s.egg' % (
        sys.version_info.major,
        sys.version_info.minor,
        'win-amd64' if os.name == 'nt' else 'linux-x86_64'))[0])
except IndexError:
    pass


#globalne zmienne
WIDTH = 1280
HEIGHT = 720
FPS = 60
FIELD_OF_VIEW = 90.0
DEFAULT_TOWN = "Town04"
METERS_PER_FRAME = 1.0
SAVE_DIR = 'data/dataset/' + DEFAULT_TOWN + '/'



def save_image(image, folder, filename):
    """Zapisuje obraz w formacie .png"""
    if not os.path.exists(folder):
        os.makedirs(folder)
    filepath = os.path.join(folder, filename)
    cv2.imwrite(filepath, image)


def image_reshape(image):
    arr = np.frombuffer(image.raw_data, dtype=np.dtype("uint8"))
    arr = np.reshape(arr, (image.height, image.width, 4))
    arr = arr[:, :, :3]
    arr = arr[:, :, ::-1]
    return arr


def image_draw(surface, image, blend=False):
    arr = image_reshape(image)
    img_surface = pygame.surfarray.make_surface(arr.swapaxes(0, 1))
    if blend:
        img_surface.set_alpha(100)
    surface.blit(img_surface, (0, 0))


def exit_game():
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            return True
        elif event.type == pygame.KEYUP:
            if event.key == pygame.K_ESCAPE:
                return True
    return False


# ==============================================================================
# -- CarlaSyncMode ------------------------------------------------------------
# ==============================================================================

class CarlaSensorSyncManager(object):
    """
    Context manager to synchronize output from different sensors. Synchronous
    mode is enabled as long as we are inside this context.

        with CarlaSyncMode(world, sensors) as sync_mode:
            while True:
                data = sync_mode.tick(timeout=1.0)

    """

    def __init__(self, world, *sensors, **kwargs):
        self.world = world
        self.sensors = sensors
        self.current_frame = None
        self.frame_interval_seconds = 1.0 / kwargs.get('fps', FPS)
        self.sensor_queues = []
        self.settings = None

    def __enter__(self):
        self.settings = self.world.get_settings()
        self.current_frame = self.world.apply_settings(carla.WorldSettings(
            no_rendering_mode=False,
            synchronous_mode=True,
            fixed_delta_seconds=self.frame_interval_seconds))

        def make_queue(register_event):
            q = queue.Queue()
            register_event(q.put)
            self.sensor_queues.append(q)

        make_queue(self.world.on_tick)
        for sensor in self.sensors:
            make_queue(sensor.listen)
        return self

    def get_sensor_data(self, sensor_queue, timeout):
        while True:
            data = sensor_queue.get(timeout=timeout)
            if data.frame == self.current_frame:
                return data

    def tick(self, timeout):
        self.current_frame = self.world.tick()
        data = [self.get_sensor_data(q, timeout) for q in self.sensor_queues]
        assert all(x.frame == self.current_frame for x in data)
        return data

    def __exit__(self, *args, **kwargs):
        self.world.apply_settings(self.settings)


# ==============================================================================
# -- Vehicle Manager ----------------------------------------------------------
# ==============================================================================

class VehicleManager:
    """
    Helper class to spawn and manage neighbor vehicles
    """

    def __init__(self):
        self.vehicles_list = []
        self.vehicleswap_counter = 0
        self.waypoint_indices = [0, 0, 0, 0, 0]  # Domyślne indeksy dla pojazdów

    def move_agent(self, vehicle, waypoint_list):
        target_waypoint = waypoint_list[0]
        vehicle.set_transform(target_waypoint.transform)
        potential_new_waypoints = waypoint_list[-1].next(METERS_PER_FRAME)
        new_waypoint = random.choice(potential_new_waypoints)
        waypoint_list.append(new_waypoint)

        return new_waypoint

    def randomize_vehicle_positions(self, waypoint_list):
        """Losowo zmienia indeksy waypointów dla sąsiednich pojazdów."""
        self.waypoint_indices = [random.randint(0, len(waypoint_list) - 1) for _ in range(len(self.vehicles_list))]

    def spawn_vehicles(self, world):
        """
        Generuje 5 losowych pojazdów aby symulować realne warunki jazdy
        """
        self.transforms = [carla.Transform(carla.Location(-1000, -1000, 0)),
                           carla.Transform(carla.Location(-1000, -1010, 0)),
                           carla.Transform(carla.Location(-1000, -1020, 0)),
                           carla.Transform(carla.Location(-1000, -1030, 0)),
                           carla.Transform(carla.Location(-1000, -1040, 0))]

        spawn_points = world.get_map().get_spawn_points()
        vehicles = world.get_blueprint_library().filter('vehicle.*')
        cars = [vehicle for vehicle in vehicles if int(vehicle.get_attribute('number_of_wheels')) == 4]
        random.shuffle(cars)

        for i, car in enumerate(cars[:5]):
            neighbor_vehicle = world.spawn_actor(car, spawn_points[i])
            neighbor_vehicle.set_simulate_physics(False)
            neighbor_vehicle.set_transform(self.transforms[i])
            self.vehicles_list.append(neighbor_vehicle)

    def move_vehicles(self, waypoint_list, frame_counter=50):
        def get_transform(waypoint):
            if waypoint and waypoint.lane_type == carla.LaneType.Driving:
                return carla.Transform(waypoint.transform.location, waypoint.transform.rotation)
            return None

        for i, vehicle in enumerate(self.vehicles_list):
            if i >= len(self.waypoint_indices):  # Zabezpieczenie przed błędnym indeksem
                continue

            lane_offset = [0, -1, 1, -2, 2][i]  # Przesunięcia na pasach
            lane = waypoint_list[self.waypoint_indices[i]].get_left_lane() if lane_offset < 0 else waypoint_list[
                self.waypoint_indices[i]].get_right_lane()
            transform = get_transform(lane)
            vehicle.set_transform(transform or self.transforms[i])

        # Losowe przełączanie pojazdów
        if self.vehicleswap_counter > frame_counter:
            self.randomize_vehicle_positions(waypoint_list)
            self.vehicleswap_counter = 0

        self.vehicleswap_counter += 1

    def destroy(self):
        for vehicle in self.vehicles_list:
            vehicle.destroy()


# ==============================================================================
# -- Carla Game ---------------------------------------------------------------
# ==============================================================================

class CarlaGame:
    def __init__(self):
        pygame.init()
        self.actors_list = []
        self.start_position = None
        self.display = pygame.display.set_mode((WIDTH, HEIGHT), pygame.HWSURFACE | pygame.DOUBLEBUF)
        self.font = pygame.font.SysFont("Arial", 14)
        self.clock = pygame.time.Clock()
        self.client = carla.Client('localhost', 2000)
        self.client.set_timeout(30.0)
        self.world = self.client.load_world(DEFAULT_TOWN)
        self.map = self.world.get_map()
        self.vehiclemanager = VehicleManager()

        self.image_counter = 0
        self.camera_transforms = [carla.Transform(carla.Location(x=0.0, z=3.2), carla.Rotation(pitch=-19.5)),
                                  carla.Transform(carla.Location(x=0.0, z=2.8), carla.Rotation(pitch=-18.5)),
                                  carla.Transform(carla.Location(x=0.3, z=2.4), carla.Rotation(pitch=-15.0)),
                                  carla.Transform(carla.Location(x=1.1, z=2.0), carla.Rotation(pitch=-16.5)),
                                  carla.Transform(carla.Location(x=1.0, z=2.0), carla.Rotation(pitch=-18.5))]

    def reset_vehicle_position(self):
        """
        Resetuje pozycje pojazdu na mapie i generuje nową trasę waypointów.
        """
        self.start_position = random.choice(self.map.get_spawn_points())
        waypoint = self.map.get_waypoint(self.start_position.location)

        self.waypoint_list = deque(maxlen=80)
        for _ in range(80):
            self.waypoint_list.append(waypoint)
            waypoint = waypoint.next(METERS_PER_FRAME)[0]

        # Dodaj przesunięcie pojazdu do pierwszego waypointa
        self.vehiclemanager.move_agent(self.vehicle, self.waypoint_list)

        camera_index = random.randint(0, len(self.camera_transforms) - 1)
        self.camera_rgb.set_transform(self.camera_transforms[camera_index])
        self.camera_semantic.set_transform(self.camera_transforms[camera_index])
        print("Camera RGB Index:", camera_index)

    def render_display(self, image, image_semseg):
        image_draw(self.display, image)
        self.display.blit(self.font.render('% 5d FPS ' % self.clock.get_fps(), True, (255, 255, 255)), (8, 10))
        self.display.blit(self.font.render('Map: ' + DEFAULT_TOWN, True, (255, 255, 255)), (20, 50))

        pygame.display.flip()

    def execute(self):
        try:
            self.initialize()
            with CarlaSensorSyncManager(self.world, self.camera_rgb, self.camera_semantic, fps=FPS) as self.sync_mode:
                while True:
                    if exit_game():
                        return
                    self.loop()

        finally:
            print('Saving files...')
            print('Destroying actors and cleaning up.')
            for actor in self.actors_list:
                actor.destroy()

            self.vehiclemanager.destroy()

            pygame.quit()
            print('Done.')

    def initialize(self):
        """
        Initialize the world and spawn vehicles, cameras and saver.
        """
        self.blueprint_library = self.world.get_blueprint_library()

        self.start_position = random.choice(self.map.get_spawn_points())
        self.vehicle = self.world.spawn_actor(random.choice(self.blueprint_library.filter('vehicle.tesla.model3')),
                                              self.start_position)
        self.actors_list.append(self.vehicle)
        self.vehicle.set_simulate_physics(False)

        self.camera_transform = self.camera_transforms[random.randint(0, len(self.camera_transforms) - 1)]

        # Spawn rgb-cam and attach to vehicle
        self.rgb_camera = self.blueprint_library.find('sensor.camera.rgb')
        self.rgb_camera.set_attribute('image_size_x', f'{WIDTH}')
        self.rgb_camera.set_attribute('image_size_y', f'{HEIGHT}')
        self.rgb_camera.set_attribute('fov', f'{FIELD_OF_VIEW}')
        self.rgb_camera_spawn = self.camera_transform
        self.camera_rgb = self.world.spawn_actor(self.rgb_camera, self.rgb_camera_spawn, attach_to=self.vehicle)
        self.actors_list.append(self.camera_rgb)

        # Spawn semseg-cam and attach to vehicle
        self.semantic_camera = self.blueprint_library.find('sensor.camera.semantic_segmentation')
        self.semantic_camera.set_attribute('image_size_x', f'{WIDTH}')
        self.semantic_camera.set_attribute('image_size_y', f'{HEIGHT}')
        self.semantic_camera.set_attribute('fov', f'{FIELD_OF_VIEW}')
        self.camera_semantic_spawnpoint = self.camera_transform
        self.camera_semantic = self.world.spawn_actor(self.semantic_camera, self.camera_semantic_spawnpoint,
                                                      attach_to=self.vehicle)
        self.actors_list.append(self.camera_semantic)

        self.reset_vehicle_position()

        # Spawn five random vehicles around the car to create a realistic traffic scenario
        self.vehiclemanager.spawn_vehicles(self.world)

    def loop(self):
        """
        Główna logika wykonywana w każdej klatce symulacji.
        """
        self.clock.tick()

        # Pobranie danych z kamer
        snapshot, image_rgb, image_semantic = self.sync_mode.tick(timeout=1.0)

        # Przesunięcie pojazdu
        new_waypoint = self.vehiclemanager.move_agent(self.vehicle, self.waypoint_list)

        # Aktualizacja waypointów (dodanie nowego punktu na końcu trasy)
        if new_waypoint:
            next_waypoint = new_waypoint.next(METERS_PER_FRAME)
            if next_waypoint:
                self.waypoint_list.append(next_waypoint[0])

        # Przemieszczenie sąsiednich pojazdów
        self.vehiclemanager.move_vehicles(self.waypoint_list)

        # Przetwarzanie obrazów semantycznych
        image_semantic.convert(carla.ColorConverter.CityScapesPalette)
        image_semantic = image_reshape(image_semantic)

        # Renderowanie obrazu
        self.render_display(image_rgb, image_semantic)

        # Zapis obrazów
        rgb_image = image_reshape(image_rgb)
        save_image(rgb_image, os.path.join(SAVE_DIR, 'train/images'), f'{self.image_counter:04d}.png')
        save_image(image_semantic, os.path.join(SAVE_DIR, 'train/masks'), f'{self.image_counter:04d}.png')

        self.image_counter += 1


def main():
    carlaGame = CarlaGame()
    carlaGame.execute()


if __name__ == '__main__':
    try:
        main()
    except KeyboardInterrupt:
        print('\nCancelled by user. Bye!')
