import carla
import torch
from torchvision import transforms
from PIL import Image
from Old.train_model import LaneDetectionModel  # Import modelu z treningu

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
MODEL_PATH = "../model_checkpoint.pth"


def load_model(model_path, input_height, input_width, num_lanes, griding_num):
    model = LaneDetectionModel(input_height, input_width, num_lanes, griding_num)
    model.load_state_dict(torch.load(model_path, map_location=DEVICE)["model_state_dict"])
    model.to(DEVICE)
    model.eval()
    return model


def preprocess_image(image, input_height, input_width):
    img_transform = transforms.Compose([
        transforms.Resize((input_height, input_width)),
        transforms.ToTensor()
    ])
    image = img_transform(image)
    image = image.unsqueeze(0)  # Dodaj wymiar batch
    return image.to(DEVICE)


def main():
    input_height = 288
    input_width = 800
    num_lanes = 4
    griding_num = 100

    # Wczytanie modelu
    model = load_model(MODEL_PATH, input_height, input_width, num_lanes, griding_num)

    # Inicjalizacja CARLA
    client = carla.Client('localhost', 2000)
    client.set_timeout(10.0)
    world = client.get_world()

    blueprint_library = world.get_blueprint_library()
    vehicle_bp = blueprint_library.filter('model3')[0]
    spawn_point = world.get_map().get_spawn_points()[0]

    vehicle = world.spawn_actor(vehicle_bp, spawn_point)
    camera_bp = blueprint_library.find('sensor.camera.rgb')
    camera_bp.set_attribute('image_size_x', str(1280))
    camera_bp.set_attribute('image_size_y', str(720))
    camera_bp.set_attribute('fov', '90')

    camera_transform = carla.Transform(carla.Location(x=1.5, z=2.4))
    camera = world.spawn_actor(camera_bp, camera_transform, attach_to=vehicle)

    def process_image(image):
        array = Image.frombytes('RGB', (image.width, image.height), image.raw_data, 'raw', 'RGB')
        processed_image = preprocess_image(array, input_height, input_width)
        with torch.no_grad():
            predictions = model(processed_image)
            print("Predictions:", torch.argmax(predictions, dim=1).cpu().numpy())

    camera.listen(lambda image: process_image(image))

    try:
        while True:
            world.wait_for_tick()
    except KeyboardInterrupt:
        print("Exiting...")
    finally:
        camera.stop()
        vehicle.destroy()


if __name__ == "__main__":
    main()
