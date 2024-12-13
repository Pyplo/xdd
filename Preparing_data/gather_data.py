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
from datetime import datetime
from camera_geometry import get_intrinsic_matrix, project_polyline, CameraGeometry
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
store_files = True
now = datetime.now()
date_time_string = now.strftime("%m_%d_%Y_%H_%M_%S")

# Funkcja pomocnicza do tworzenia katalogów
def ensure_dir_exists(filepath):
    directory = os.path.dirname(filepath)
    if not os.path.exists(directory):
        os.makedirs(directory)

def carla_vec_to_np_array(location):
    """
    Konwertuje obiekt carla.Location do tablicy NumPy.
    """
    return np.array([location.x, location.y, location.z])


def plot_map(m):
    import matplotlib.pyplot as plt

    wp_list = m.generate_waypoints(2.0)
    loc_list = np.array(
        [carla_vec_to_np_array(wp.transform.location) for wp in wp_list]
    )
    plt.scatter(loc_list[:, 0], loc_list[:, 1])
    plt.show()


def random_transform_disturbance(transform):
    lateral_noise = np.random.normal(0, 0.3)
    lateral_noise = np.clip(lateral_noise, -0.3, 0.3)

    lateral_direction = transform.get_right_vector()
    x = transform.location.x + lateral_noise * lateral_direction.x
    y = transform.location.y + lateral_noise * lateral_direction.y
    z = transform.location.z + lateral_noise * lateral_direction.z

    yaw_noise = np.random.normal(0, 5)
    yaw_noise = np.clip(yaw_noise, -10, 10)

    pitch = transform.rotation.pitch
    yaw = transform.rotation.yaw + yaw_noise
    roll = transform.rotation.roll

    return carla.Transform(
        carla.Location(x, y, z), carla.Rotation(pitch, yaw, roll)
    )


def get_curvature(polyline):
    dx_dt = np.gradient(polyline[:, 0])
    dy_dt = np.gradient(polyline[:, 1])
    d2x_dt2 = np.gradient(dx_dt)
    d2y_dt2 = np.gradient(dy_dt)
    curvature = (
            np.abs(d2x_dt2 * dy_dt - dx_dt * d2y_dt2)
            / (dx_dt * dx_dt + dy_dt * dy_dt) ** 1.5
    )
    # print(curvature)
    return np.max(curvature)


def create_lane_lines(
    world_map, vehicle, exclude_junctions=True, only_turns=False
):
    waypoint = world_map.get_waypoint(
        vehicle.get_transform().location,
        project_to_road=True,
        lane_type=carla.LaneType.Driving,
    )
    # print(str(waypoint.right_lane_marking.type))
    center_list, left_boundary, right_boundary = [], [], []
    for _ in range(60):
        if (
            str(waypoint.right_lane_marking.type)
            + str(waypoint.left_lane_marking.type)
        ).find("NONE") != -1:
            return None, None, None
        # if there is a junction on the path, return None
        if exclude_junctions and waypoint.is_junction:
            return None, None, None
        next_waypoints = waypoint.next(1.0)
        # if there is a branch on the path, return None
        if len(next_waypoints) != 1:
            return None, None, None
        waypoint = next_waypoints[0]
        center = carla_vec_to_np_array(waypoint.transform.location)
        center_list.append(center)
        offset = (
            carla_vec_to_np_array(waypoint.transform.get_right_vector())
            * waypoint.lane_width
            / 2.0
        )
        left_boundary.append(center - offset)
        right_boundary.append(center + offset)

    max_curvature = get_curvature(np.array(center_list))
    if max_curvature > 0.005:
        return None, None, None

    if only_turns and max_curvature < 0.002:
        return None, None, None

    return (
        np.array(center_list),
        np.array(left_boundary),
        np.array(right_boundary),
    )







def check_inside_image(pixel_array, width, height):
    ok = (0 < pixel_array[:, 0]) & (pixel_array[:, 0] < width)
    ok = ok & (0 < pixel_array[:, 1]) & (pixel_array[:, 1] < height)
    ratio = np.sum(ok) / len(pixel_array)
    return ratio > 0.5


def save_image(image, path, raw=False):
    array = image_reshape(image)
    if raw:
        np.save(path, array)
    else:
        cv2.imwrite(path, array)


def save_label_img(lb_left, lb_right, path):
    label = np.zeros((HEIGHT, WIDTH, 3))
    colors = [[1, 1, 1], [2, 2, 2]]
    for color, lb in zip(colors, [lb_left, lb_right]):
        cv2.polylines(
            label, np.int32([lb]), isClosed=False, color=color, thickness=5
        )
    label = np.mean(label, axis=2)  # collapse color channels to get gray scale
    cv2.imwrite(path, label)


def image_reshape(image):
    arr = np.frombuffer(image.raw_data, dtype=np.dtype("uint8"))
    arr = np.reshape(arr, (image.height, image.width, 4))
    arr = arr[:, :, :3]
    arr = arr[:, :, ::-1]
    return arr


def get_random_spawn_point(m):
    pose = random.choice(m.get_spawn_points())
    return m.get_waypoint(pose.location)




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
        self.min_jump, self.max_jump = 5, 10
        self.frame = 0
        self.vehiclemanager = VehicleManager()
        self.spawn_waypoint = None
        self.image_counter = 0

    def reset_vehicle_position(self):
        """
        Resetuje pozycje pojazdu na mapie i generuje nową trasę waypointów.
        """
        self.start_position = get_random_spawn_point(self.map).transform
        waypoint = self.map.get_waypoint(self.start_position.location)

        self.waypoint_list = deque(maxlen=80)
        for _ in range(80):
            self.waypoint_list.append(waypoint)
            waypoint = waypoint.next(METERS_PER_FRAME)[0]

        # Dodaj przesunięcie pojazdu do pierwszego waypointa
        self.vehiclemanager.move_agent(self.vehicle, self.waypoint_list)

        camera_index = random.randint(0, len(self.camera_transforms) - 1)
        disturbed_transform = random_transform_disturbance(self.camera_transforms[camera_index])
        self.camera_rgb.set_transform(disturbed_transform)
        self.camera_semantic.set_transform(disturbed_transform)

        print("Camera RGB Index:", camera_index)

    def render_display(self, image, image_semseg=None):
        """
        Renderuje obraz i informacje na ekranie.
        """
        image_draw(self.display, image)
        self.display.blit(
            self.font.render('% 5d FPS (real)' % self.clock.get_fps(), True, (255, 255, 255)),
            (8, 10)
        )
        if hasattr(self, 'snapshot'):
            fps = round(1.0 / self.snapshot.timestamp.delta_seconds)
            self.display.blit(
                self.font.render('% 5d FPS (simulated)' % fps, True, (255, 255, 255)),
                (8, 28)
            )
        pygame.display.flip()

    def initialize(self):
        """
        Inicjalizacja świata, pojazdu, kamery oraz waypointów.
        """
        self.blueprint_library = self.world.get_blueprint_library()
        self.start_position = random.choice(self.map.get_spawn_points())
        self.vehicle = self.world.spawn_actor(
            random.choice(self.blueprint_library.filter('vehicle.audi.tt')),
            self.start_position
        )
        if self.vehicle is None:
            raise RuntimeError("Nie udało się zespawnować pojazdu.")
        self.actors_list.append(self.vehicle)
        self.vehicle.set_simulate_physics(False)

        self.spawn_waypoint = self.map.get_waypoint(self.start_position.location)

        # Kamera RGB
        cg = CameraGeometry()
        cam_rgb_transform = carla.Transform(
            carla.Location(x=0.5, z=cg.height),
            carla.Rotation(pitch=cg.pitch_deg)
        )
        self.trafo_matrix_vehicle_to_cam = np.array(cam_rgb_transform.get_inverse_matrix())
        bp = self.blueprint_library.find('sensor.camera.rgb')
        bp.set_attribute('image_size_x', f'{WIDTH}')
        bp.set_attribute('image_size_y', f'{HEIGHT}')
        bp.set_attribute('fov', f'{cg.field_of_view_deg}')
        self.camera_rgb = self.world.spawn_actor(bp, cam_rgb_transform, attach_to=self.vehicle)
        if self.camera_rgb is None:
            raise RuntimeError("Nie udało się zespawnować kamery RGB.")
        self.actors_list.append(self.camera_rgb)

        self.K = get_intrinsic_matrix(cg.field_of_view_deg, WIDTH, HEIGHT)

    def loop(self):
        try:
            self.snapshot, image_rgb = self.sync_mode.tick(timeout=2.0)

            # Linie pasów
            center_list, left_boundary, right_boundary = create_lane_lines(self.map, self.vehicle)

            if center_list is None or left_boundary is None or right_boundary is None:
                print("Brak danych linii pasów. Reset waypoint.")
                self.spawn_waypoint = get_random_spawn_point(self.map)
                return

            # Użyjmy bezpiecznego sprawdzenia, aby uniknąć próby subskrypcji None
            if center_list is not None and left_boundary is not None and right_boundary is not None:
                # Projektowanie linii na obraz (prosta projekcja, tylko do wyświetlania)
                trafo_matrix_world_to_vehicle = np.array(self.vehicle.get_transform().get_inverse_matrix())
                trafo_matrix_global_to_camera = self.trafo_matrix_vehicle_to_cam @ trafo_matrix_world_to_vehicle
                mat_swap_axes = np.array([[0, 1, 0, 0], [0, 0, -1, 0], [1, 0, 0, 0], [0, 0, 0, 1]])
                trafo_matrix_global_to_camera = mat_swap_axes @ trafo_matrix_global_to_camera

                projected_center = project_polyline(center_list, trafo_matrix_global_to_camera, self.K).astype(np.int32)
                projected_left_boundary = project_polyline(left_boundary, trafo_matrix_global_to_camera, self.K).astype(
                    np.int32)
                projected_right_boundary = project_polyline(right_boundary, trafo_matrix_global_to_camera,
                                                            self.K).astype(np.int32)

                # Rysowanie linii na obrazie
                if len(projected_center) > 1:
                    pygame.draw.lines(self.display, (255, 136, 0), False, projected_center, 4)
                if len(projected_left_boundary) > 1:
                    pygame.draw.lines(self.display, (255, 0, 0), False, projected_left_boundary, 4)
                if len(projected_right_boundary) > 1:
                    pygame.draw.lines(self.display, (0, 255, 0), False, projected_right_boundary, 4)

                # Wyświetlanie obrazu

                self.render_display(image_rgb)
                # Aktualizacja licznika klatek
                self.frame += 1
            else:
                print("Brak danych dla linii pasów, próbuję zresetować waypoint.")
                self.spawn_waypoint = get_random_spawn_point(self.map)

        except Exception as e:
            print(f"Błąd w loop: {e}")

    def execute(self):
        try:
            self.initialize()
            with CarlaSensorSyncManager(self.world, self.camera_rgb, fps=FPS) as self.sync_mode:
                while True:
                    if exit_game():
                        return
                    self.loop()

        finally:
            print('Cleaning up...')
            for actor in self.actors_list:
                actor.destroy()
            self.vehiclemanager.destroy()
            self.world.apply_settings(carla.WorldSettings(no_rendering_mode=False, synchronous_mode=False))
            pygame.quit()
def main():
    carlaGame = CarlaGame()
    carlaGame.execute()


if __name__ == '__main__':
    try:
        main()
    except KeyboardInterrupt:
        print('\nCancelled by user. Bye!')
