import torch
from torch.utils.data import Dataset
from torchvision import transforms
import json
from PIL import Image
import os

class LaneDataset(Dataset):
    def __init__(self, root_dir, json_file, input_height, input_width):
        """
        Args:
            root_dir (str): Główny katalog z danymi.
            json_file (str): Ścieżka do pliku JSON z etykietami.
            input_height (int): Wysokość obrazów po przeskalowaniu.
            input_width (int): Szerokość obrazów po przeskalowaniu.
        """
        self.root_dir = root_dir
        self.input_height = input_height
        self.input_width = input_width

        with open(os.path.join(root_dir, json_file), 'r') as f:
            self.labels = [json.loads(line) for line in f]

        self.transforms = transforms.Compose([
            transforms.Resize((input_height, input_width)),
            transforms.ToTensor()
        ])

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        # Pobierz dane z JSON-a
        label_data = self.labels[idx]
        image_path = os.path.join(self.root_dir, label_data['raw_file'])
        lanes = label_data['lanes']
        h_samples = label_data['h_samples']

        # Załaduj obraz
        image = Image.open(image_path).convert("RGB")
        image = self.transforms(image)

        # Konwersja etykiety na tensor
        label = self._process_labels(lanes, h_samples)
        return image, label

    def _process_labels(self, lanes, h_samples):
        """
        Przetwarzanie etykiet z pliku JSON do postaci tensorów.
        """
        grid_labels = torch.full((len(h_samples), len(lanes)), -1, dtype=torch.float32)

        for i, lane in enumerate(lanes):
            for j, x in enumerate(lane):
                if x != -2:  # -2 oznacza brak punktu
                    grid_labels[j, i] = x
        return grid_labels
