import os
import carla
import random
import pygame
import numpy as np
import cv2
from datetime import datetime

from camera_geometry import get_intrinsic_matrix, project_polyline, CameraGeometry

# Constants
STORE_FILES = True
TOWN_NAME = "Town04"
CAMERA_GEOMETRY = CameraGeometry()
WIDTH, HEIGHT = CAMERA_GEOMETRY.image_width, CAMERA_GEOMETRY.image_height
DATE_TIME_STRING = datetime.now().strftime("%m_%d_%Y_%H_%M_%S")
data_folder = 'C:/Users/Patry/Desktop/FinalProject'


def carla_vec_to_np_array(location):
    """
    Konwertuje obiekt carla.Location do tablicy NumPy.
    """
    return np.array([location.x, location.y, location.z])


# Helper Functions
def plot_map(world_map):
    import matplotlib.pyplot as plt

    wp_list = world_map.generate_waypoints(2.0)
    loc_list = np.array([
        carla_vec_to_np_array(wp.transform.location) for wp in wp_list
    ])
    plt.scatter(loc_list[:, 0], loc_list[:, 1])
    plt.show()


def random_transform_disturbance(transform):
    lateral_noise = np.clip(np.random.normal(0, 0.3), -0.3, 0.3)
    lateral_direction = transform.get_right_vector()
    location = transform.location
    location.x += lateral_noise * lateral_direction.x
    location.y += lateral_noise * lateral_direction.y
    location.z += lateral_noise * lateral_direction.z

    yaw_noise = np.clip(np.random.normal(0, 5), -10, 10)
    rotation = transform.rotation
    rotation.yaw += yaw_noise

    return carla.Transform(location, rotation)


def get_curvature(polyline):
    dx_dt = np.gradient(polyline[:, 0])
    dy_dt = np.gradient(polyline[:, 1])
    d2x_dt2 = np.gradient(dx_dt)
    d2y_dt2 = np.gradient(dy_dt)
    curvature = (
            np.abs(d2x_dt2 * dy_dt - dx_dt * d2y_dt2)
            / (dx_dt ** 2 + dy_dt ** 2) ** 1.5
    )
    return np.max(curvature)


def create_lane_lines(world_map, vehicle, exclude_junctions=True, only_turns=False):
    waypoint = world_map.get_waypoint(
        vehicle.get_transform().location,
        project_to_road=True,
        lane_type=carla.LaneType.Driving,
    )

    center_list, left_boundary, right_boundary = [], [], []
    for _ in range(60):
        if "NONE" in f"{waypoint.right_lane_marking.type}{waypoint.left_lane_marking.type}":
            return None, None, None
        if exclude_junctions and waypoint.is_junction:
            return None, None, None
        next_waypoints = waypoint.next(1.0)
        if len(next_waypoints) != 1:
            return None, None, None

        waypoint = next_waypoints[0]
        center = carla_vec_to_np_array(waypoint.transform.location)
        center_list.append(center)

        offset = carla_vec_to_np_array(waypoint.transform.get_right_vector()) * waypoint.lane_width / 2.0
        left_boundary.append(center - offset)
        right_boundary.append(center + offset)

    max_curvature = get_curvature(np.array(center_list))
    if max_curvature > 0.005 or (only_turns and max_curvature < 0.002):
        return None, None, None

    return np.array(center_list), np.array(left_boundary), np.array(right_boundary)


def check_inside_image(pixel_array, width, height):
    valid_pixels = (0 < pixel_array[:, 0]) & (pixel_array[:, 0] < width) & (0 < pixel_array[:, 1]) & (
                pixel_array[:, 1] < height)
    return np.sum(valid_pixels) / len(pixel_array) > 0.5


def carla_img_to_array(image):
    array = np.frombuffer(image.raw_data, dtype=np.uint8)
    array = array.reshape((image.height, image.width, 4))[:, :, :3]
    return array[:, :, ::-1]


def save_img(image, path, raw=False):
    array = carla_img_to_array(image)
    if raw:
        np.save(path, array)
    else:
        cv2.imwrite(path, array)


def save_label_img(lb_left, lb_right, path):
    label = np.zeros((HEIGHT, WIDTH, 3))
    for color, lb in zip([[1, 1, 1], [2, 2, 2]], [lb_left, lb_right]):
        cv2.polylines(label, [np.int32(lb)], isClosed=False, color=color, thickness=5)
    cv2.imwrite(path, np.mean(label, axis=2))


def get_random_spawn_point(world_map):
    return random.choice(world_map.get_spawn_points()).location


def ensure_dir_exists(filepath):
    directory = os.path.dirname(filepath)
    if not os.path.exists(directory):
        os.makedirs(directory)


# Main Function
def draw_image(surface, image):
    array = np.frombuffer(image.raw_data, dtype=np.uint8)
    array = array.reshape((image.height, image.width, 4))
    array = array[:, :, :3]
    array = array[:, :, ::-1]
    surface.blit(pygame.surfarray.make_surface(array.swapaxes(0, 1)), (0, 0))


def draw_fps(display, font, clock, simulated_fps):
    display.blit(
        font.render("% 5d FPS (real)" % clock.get_fps(), True, (255, 255, 255)),
        (8, 10),
    )
    display.blit(
        font.render("% 5d FPS (simulated)" % simulated_fps, True, (255, 255, 255)),
        (8, 28),
    )


def process_lane_boundaries(world_map, vehicle, K, trafo_matrix):
    center_list, left_boundary, right_boundary = create_lane_lines(world_map, vehicle)
    if center_list is None:
        return None, None, None

    projected_center = project_polyline(center_list, trafo_matrix, K).astype(np.int32)
    projected_left_boundary = project_polyline(left_boundary, trafo_matrix, K).astype(np.int32)
    projected_right_boundary = project_polyline(right_boundary, trafo_matrix, K).astype(np.int32)

    if (
            not check_inside_image(projected_left_boundary, WIDTH, HEIGHT)
            or not check_inside_image(projected_right_boundary, WIDTH, HEIGHT)
    ):
        return None, None, None

    return projected_center, projected_left_boundary, projected_right_boundary


def save_simulation_data(data_folder, simulation_identifier, frame, image_rgb, left_boundary, right_boundary,
                         trafo_matrix, height, width, validation_ratio=0.1):
    """
    Zapisuje dane symulacji: obraz, granice, macierze transformacji.

    :param data_folder: Folder, w którym będą przechowywane dane.
    :param simulation_identifier: Unikalny identyfikator symulacji.
    :param frame: Numer ramki symulacji.
    :param image_rgb: Obraz RGB uzyskany z symulacji.
    :param left_boundary: Granica lewej linii pasa.
    :param right_boundary: Granica prawej linii pasa.
    :param trafo_matrix: Macierz transformacji globalnej do kamery.
    :param height: Wysokość obrazu.
    :param width: Szerokość obrazu.
    :param validation_ratio: Procent danych przeznaczonych do walidacji.
    """
    # Określenie, czy zapisujemy dane do zestawu walidacyjnego
    in_validation_set = np.random.rand() < validation_ratio
    subset = "validation_set" if in_validation_set else "training_set"

    # Tworzenie folderów docelowych
    subset_folder = os.path.join(data_folder, subset)
    ensure_dir_exists(subset_folder)

    # Bazowa nazwa pliku
    filename_base = os.path.join(subset_folder, f"{simulation_identifier}_frame_{frame}")

    # Zapis obrazu
    image_out_path = f"{filename_base}.png"
    save_img(image_rgb, image_out_path)

    # Zapis obrazu etykiet
    label_path = f"{filename_base}_label.png"
    save_label_img(left_boundary, right_boundary, label_path)

    # Zapis granic
    border_array = np.hstack((left_boundary, right_boundary))
    border_path = f"{filename_base}_boundary.txt"
    np.savetxt(border_path, border_array)

    # Zapis macierzy transformacji
    trafo_path = f"{filename_base}_trafo.txt"
    np.savetxt(trafo_path, trafo_matrix)

    print(f"Saved frame {frame} data to {subset_folder}")


class CarlaSyncMode:
    def __init__(self, world, *sensors, fps=30):
        self.world = world
        self.sensors = sensors
        self.delta_seconds = 1.0 / fps
        self.queue = []
        self._settings = None

    def __enter__(self):
        self._settings = self.world.get_settings()
        settings = carla.WorldSettings()
        settings.fixed_delta_seconds = self.delta_seconds
        settings.synchronous_mode = True
        self.world.apply_settings(settings)

        for sensor in self.sensors:
            sensor.listen(lambda data: self.queue.append(data))

        return self

    def tick(self, timeout=2.0):
        self.world.tick()
        data = []
        for sensor in self.sensors:
            if self.queue:
                data.append(self.queue.pop(0))
            else:
                data.append(None)

        if not all(data):
            raise RuntimeError("Nie udało się pobrać danych z sensorów.")
        return (self.world.get_snapshot(), *data)

    def __exit__(self, *args, **kwargs):
        self.world.apply_settings(self._settings)
        for sensor in self.sensors:
            sensor.stop()
        self.queue.clear()



def should_quit():
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            return True
    return False


def update_spawn_waypoint(world_map, current_waypoint, min_jump, max_jump):
    jump = np.random.uniform(min_jump, max_jump)
    next_waypoints = current_waypoint.next(jump)
    if not next_waypoints:
        return random.choice(world_map.get_spawn_points())
    else:
        return random.choice(next_waypoints)


def main():
    ensure_dir_exists(data_folder)
    pygame.init()
    display = pygame.display.set_mode((WIDTH, HEIGHT), pygame.HWSURFACE | pygame.DOUBLEBUF)
    font = pygame.font.SysFont("monospace", 12)
    clock = pygame.time.Clock()

    client = carla.Client("localhost", 2000)
    client.set_timeout(60.0)
    client.load_world(TOWN_NAME)

    world = client.get_world()
    actor_list = []

    try:
        world_map = world.get_map()
        start_pose = random.choice(world_map.get_spawn_points())
        spawn_waypoint = world_map.get_waypoint(start_pose.location)

        blueprint_library = world.get_blueprint_library()
        vehicle = world.spawn_actor(random.choice(blueprint_library.filter("vehicle.audi.tt")), start_pose)
        actor_list.append(vehicle)

        cam_transform = carla.Transform(
            carla.Location(x=0.5, z=CAMERA_GEOMETRY.height),
            carla.Rotation(pitch=CAMERA_GEOMETRY.pitch_deg),
        )
        print("Inicjalizacja kamery...")
        camera_bp = blueprint_library.find("sensor.camera.rgb")
        camera_bp.set_attribute("image_size_x", str(WIDTH))
        camera_bp.set_attribute("image_size_y", str(HEIGHT))
        camera_bp.set_attribute("fov", str(CAMERA_GEOMETRY.field_of_view_deg))

        camera_rgb = world.spawn_actor(camera_bp, cam_transform, attach_to=vehicle)
        print("Kamera została utworzona:", camera_rgb is not None)

        camera_rgb.listen(lambda data: print("Odebrano dane z kamery", data.frame))

        intrinsic_matrix = get_intrinsic_matrix(CAMERA_GEOMETRY.field_of_view_deg, WIDTH, HEIGHT)

        with CarlaSyncMode(world, camera_rgb, fps=30) as sync_mode:
            frame = 0
            while True:
                if should_quit():
                    break

                clock.tick()
                snapshot, image_rgb = sync_mode.tick(timeout=2.0)

                # Update vehicle position
                spawn_waypoint = update_spawn_waypoint(world_map, spawn_waypoint)
                spawn_transform = random_transform_disturbance(spawn_waypoint.transform)
                vehicle.set_transform(spawn_transform)

                # Draw the display
                fps = round(1.0 / snapshot.timestamp.delta_seconds)
                draw_image(display, image_rgb)
                draw_fps(display, clock.get_fps(), fps, font)

                center_list, left_boundary, right_boundary = create_lane_lines(world_map, vehicle)
                if center_list is None:
                    continue

                process_lane_boundaries(
                    display, center_list, left_boundary, right_boundary, WIDTH, HEIGHT,
                    CAMERA_GEOMETRY, intrinsic_matrix, vehicle
                )

                if STORE_FILES:
                    save_simulation_data(
                        data_folder, frame, DATE_TIME_STRING, center_list,
                        left_boundary, right_boundary, image_rgb
                    )

                pygame.display.flip()
                frame += 1

    finally:
        for actor in actor_list:
            actor.destroy()
        pygame.quit()


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\nCancelled by user. Bye!")
