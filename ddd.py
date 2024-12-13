# Code based on Carla examples, which are authored by
# Computer Vision Center (CVC) at the Universitat Autonoma de Barcelona (UAB).

# How to run:
# Start a Carla simulation
# cd into the parent directory of the 'code' directory and run
# python -m code.solutions.lane_detection.collect_data

import os
import carla
import random
import pygame
import numpy as np
import cv2
from datetime import datetime
import queue


from Preparing_data.camera_geometry import (
    get_intrinsic_matrix,
    project_polyline,
    CameraGeometry,
)



store_files = True
town_string = "Town04"
cg = CameraGeometry()
width = cg.image_width
height = cg.image_height

now = datetime.now()
date_time_string = now.strftime("%m_%d_%Y_%H_%M_%S")
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


def carla_img_to_array(image):
    array = np.frombuffer(image.raw_data, dtype=np.dtype("uint8"))
    array = np.reshape(array, (image.height, image.width, 4))
    array = array[:, :, :3]
    array = array[:, :, ::-1]
    return array

def image_draw(surface, image, blend=False):
    arr = carla_img_to_array(image)
    img_surface = pygame.surfarray.make_surface(arr.swapaxes(0, 1))
    if blend:
        img_surface.set_alpha(100)
    surface.blit(img_surface, (0, 0))
def save_img(image, path, raw=False):
    array = carla_img_to_array(image)
    if raw:
        np.save(path, array)
    else:
        cv2.imwrite(path, array)


def save_label_img(lb_left, lb_right, path):
    label = np.zeros((height, width, 3))
    colors = [[1, 1, 1], [2, 2, 2]]
    for color, lb in zip(colors, [lb_left, lb_right]):
        cv2.polylines(
            label, np.int32([lb]), isClosed=False, color=color, thickness=5
        )
    label = np.mean(label, axis=2)  # collapse color channels to get gray scale
    cv2.imwrite(path, label)


def get_random_spawn_point(m):
    pose = random.choice(m.get_spawn_points())
    return m.get_waypoint(pose.location)


data_folder = 'C:/Users/Patry/Desktop/FinalProject/sperma'
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
        self.frame_interval_seconds = 1.0 / kwargs.get('fps')
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


def ensure_dir_exists(filepath):
    directory = os.path.dirname(filepath)
    if not os.path.exists(directory):
        os.makedirs(directory)


def main():
    ensure_dir_exists(data_folder)
    actor_list = []
    pygame.init()

    display = pygame.display.set_mode(
        (width, height), pygame.HWSURFACE | pygame.DOUBLEBUF
    )
    font = pygame.font.SysFont("monospace", 12)
    clock = pygame.time.Clock()

    client = carla.Client("localhost", 2000)
    client.set_timeout(60.0)

    client.load_world(town_string)
    world = client.get_world()

    try:
        m = world.get_map()
        # plot_map(m)
        start_pose = random.choice(m.get_spawn_points())
        spawn_waypoint = m.get_waypoint(start_pose.location)

        # set weather to sunny

        simulation_identifier = (
            town_string + "_"  + "_" + date_time_string
        )

        # create a vehicle
        blueprint_library = world.get_blueprint_library()

        vehicle = world.spawn_actor(
            random.choice(blueprint_library.filter("vehicle.audi.tt")),
            start_pose,
        )
        actor_list.append(vehicle)
        vehicle.set_simulate_physics(False)

        # create camera and attach to vehicle
        cam_rgb_transform = carla.Transform(
            carla.Location(x=0.5, z=cg.height),
            carla.Rotation(pitch=cg.pitch_deg),
        )
        trafo_matrix_vehicle_to_cam = np.array(
            cam_rgb_transform.get_inverse_matrix()
        )
        bp = blueprint_library.find("sensor.camera.rgb")
        fov = cg.field_of_view_deg
        bp.set_attribute("image_size_x", str(width))
        bp.set_attribute("image_size_y", str(height))
        bp.set_attribute("fov", str(fov))
        camera_rgb = world.spawn_actor(
            bp, cam_rgb_transform, attach_to=vehicle
        )
        actor_list.append(camera_rgb)

        K = get_intrinsic_matrix(fov, width, height)
        min_jump, max_jump = 5, 10

        # Create a synchronous mode context.
        with CarlaSensorSyncManager(world, camera_rgb, fps=30) as sync_mode:
            frame = 0
            while True:
                clock.tick()

                # Advance the simulation and wait for the data.
                snapshot, image_rgb = sync_mode.tick(timeout=2.0)

                # Choose the next spawn_waypoint and update the car location.
                # ----- change lane with low probability
                if np.random.rand() > 0.9:
                    shifted = None
                    if spawn_waypoint.lane_change == carla.LaneChange.Left:
                        shifted = spawn_waypoint.get_left_lane()
                    elif spawn_waypoint.lane_change == carla.LaneChange.Right:
                        shifted = spawn_waypoint.get_right_lane()
                    elif spawn_waypoint.lane_change == carla.LaneChange.Both:
                        if np.random.rand() > 0.5:
                            shifted = spawn_waypoint.get_right_lane()
                        else:
                            shifted = spawn_waypoint.get_left_lane()
                    if shifted is not None:
                        spawn_waypoint = shifted
                # ----- jump forwards a random distance
                jump = np.random.uniform(min_jump, max_jump)
                next_waypoints = spawn_waypoint.next(jump)
                if not next_waypoints:
                    spawn_waypoint = get_random_spawn_point(m)
                else:
                    spawn_waypoint = random.choice(next_waypoints)

                # ----- randomly change yaw and lateral position
                spawn_transform = random_transform_disturbance(
                    spawn_waypoint.transform
                )
                vehicle.set_transform(spawn_transform)

                # Draw the display.
                fps = round(1.0 / snapshot.timestamp.delta_seconds)

                image_draw(display, image_rgb)
                display.blit(
                    font.render(
                        "% 5d FPS (real)" % clock.get_fps(),
                        True,
                        (255, 255, 255),
                    ),
                    (8, 10),
                )
                display.blit(
                    font.render(
                        "% 5d FPS (simulated)" % fps, True, (255, 255, 255)
                    ),
                    (8, 28),
                )

                # draw lane boundaries as augmented reality
                trafo_matrix_world_to_vehicle = np.array(
                    vehicle.get_transform().get_inverse_matrix()
                )
                trafo_matrix_global_to_camera = (
                    trafo_matrix_vehicle_to_cam @ trafo_matrix_world_to_vehicle
                )
                mat_swap_axes = np.array(
                    [[0, 1, 0, 0], [0, 0, -1, 0], [1, 0, 0, 0], [0, 0, 0, 1]]
                )
                trafo_matrix_global_to_camera = (
                    mat_swap_axes @ trafo_matrix_global_to_camera
                )

                center_list, left_boundary, right_boundary = create_lane_lines(
                    m, vehicle
                )
                if center_list is None:
                    spawn_waypoint = get_random_spawn_point(m)
                    continue

                projected_center = project_polyline(
                    center_list, trafo_matrix_global_to_camera, K
                ).astype(np.int32)
                projected_left_boundary = project_polyline(
                    left_boundary, trafo_matrix_global_to_camera, K
                ).astype(np.int32)
                projected_right_boundary = project_polyline(
                    right_boundary, trafo_matrix_global_to_camera, K
                ).astype(np.int32)
                if (
                    not check_inside_image(
                        projected_right_boundary, width, height
                    )
                ) or (
                    not check_inside_image(
                        projected_right_boundary, width, height
                    )
                ):
                    spawn_waypoint = get_random_spawn_point(m)
                    continue
                if len(projected_center) > 1:
                    pygame.draw.lines(
                        display, (255, 136, 0), False, projected_center, 4
                    )
                if len(projected_left_boundary) > 1:
                    pygame.draw.lines(
                        display, (255, 0, 0), False, projected_left_boundary, 4
                    )
                if len(projected_right_boundary) > 1:
                    pygame.draw.lines(
                        display,
                        (0, 255, 0),
                        False,
                        projected_right_boundary,
                        4,
                    )

                in_lower_part_of_map = spawn_transform.location.y < 0

                if store_files:
                    filename_base = simulation_identifier + "_frame_{}".format(
                        frame
                    )
                    if in_lower_part_of_map:
                        if (
                            np.random.rand() > 0.1
                        ):  # do not need that many files from validation set
                            continue
                        filename_base += "_validation_set"
                    # image
                    image_out_path = os.path.join(
                        data_folder, filename_base + ".png"
                    )
                    save_img(image_rgb, image_out_path)
                    # label img
                    label_path = os.path.join(
                        data_folder, filename_base + "_label.png"
                    )
                    save_label_img(
                        projected_left_boundary,
                        projected_right_boundary,
                        label_path,
                    )
                    # borders
                    border_array = np.hstack(
                        (np.array(left_boundary), np.array(right_boundary))
                    )
                    border_path = os.path.join(
                        data_folder, filename_base + "_boundary.txt"
                    )
                    np.savetxt(border_path, border_array)
                    # trafo
                    trafo_path = os.path.join(
                        data_folder, filename_base + "_trafo.txt"
                    )
                    np.savetxt(trafo_path, trafo_matrix_global_to_camera)

                curvature = get_curvature(center_list)
                if curvature > 0.0005:
                    min_jump, max_jump = 1, 2
                else:
                    min_jump, max_jump = 5, 10

                pygame.display.flip()
                frame += 1

    finally:

        print("destroying actors.")
        for actor in actor_list:
            actor.destroy()

        pygame.quit()
        print("done.")


if __name__ == "__main__":

    try:

        main()

    except KeyboardInterrupt:
        print("\nCancelled by user. Bye!")
