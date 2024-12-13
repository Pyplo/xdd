import os
import warnings
import cv2
import tqdm
import numpy as np
import json


def calc_k(line):
    """
    Calculate the direction of lanes.
    """
    line_x = line[::2]
    line_y = line[1::2]
    length = np.sqrt((line_x[0] - line_x[-1]) ** 2 + (line_y[0] - line_y[-1]) ** 2)
    if length < 90:
        return -10  # Skip lanes that are too short

    p = np.polyfit(line_x, line_y, deg=1)
    rad = np.arctan(p[0])
    return rad


def draw(im, line, idx):
    """
    Generate the segmentation label according to JSON annotation.
    """
    line_x = line[::2]
    line_y = line[1::2]
    pt0 = (int(line_x[0]), int(line_y[0]))

    for i in range(len(line_x) - 1):
        cv2.line(im, pt0, (int(line_x[i + 1]), int(line_y[i + 1])), (idx,), thickness=16)
        pt0 = (int(line_x[i + 1]), int(line_y[i + 1]))


def process_json_file(root, json_file, output_seg_dir, train_gt_file):
    """
    Process a JSON file and generate segmentation masks and training index file.
    """
    # Create output directories if not exist
    os.makedirs(output_seg_dir, exist_ok=True)

    # Load the JSON annotations
    with open(json_file, 'r') as file:
        label_json = [json.loads(line) for line in file.readlines()]

    # Prepare output training file
    train_gt_fp = open(train_gt_file, 'w')

    for annotation in tqdm.tqdm(label_json):
        # Extract data from JSON
        lanes = annotation['lanes']
        h_samples = np.array(annotation['h_samples'])
        raw_file = annotation['raw_file']

        # Generate label image
        label_path = os.path.join(output_seg_dir, os.path.basename(raw_file).replace('.jpg', '.png'))
        label = np.zeros((720, 1280), dtype=np.uint8)
        lines = []

        # Prepare lane lines
        for lane in lanes:
            if np.all(np.array(lane) == -2):
                continue
            valid = np.array(lane) != -2
            valid_length = np.count_nonzero(valid)  # liczba prawidłowych punktów
            line = np.zeros(valid_length * 2, dtype=np.float32)  # dynamicznie tworzymy tablicę
            line[::2] = np.array(lane)[valid]
            line[1::2] = h_samples[valid]
            lines.append(line)

        # Calculate directions and sort lanes
        with warnings.catch_warnings():
            warnings.filterwarnings('error')
            try:
                ks = np.array([calc_k(line) for line in lines])
            except np.RankWarning:
                print(f"Rank warning in file {raw_file}", flush=True)
                continue

        k_neg = ks[ks < 0]
        k_pos = ks[ks > 0]
        k_neg = k_neg[k_neg != -10]
        k_pos = k_pos[k_pos != -10]
        k_neg.sort()
        k_pos.sort()

        bin_label = [0, 0, 0, 0]

        # Assign lanes to segmentation labels
        if len(k_neg) >= 1:
            draw(label, lines[np.where(ks == k_neg[0])[0][0]], 2)
            bin_label[1] = 1
        if len(k_neg) >= 2:
            draw(label, lines[np.where(ks == k_neg[1])[0][0]], 1)
            bin_label[0] = 1
        if len(k_pos) >= 1:
            draw(label, lines[np.where(ks == k_pos[0])[0][0]], 3)
            bin_label[2] = 1
        if len(k_pos) >= 2:
            draw(label, lines[np.where(ks == k_pos[-1])[0][0]], 4)
            bin_label[3] = 1

        # Save the label image
        cv2.imwrite(label_path, label)

        # Write to training index file
        train_gt_fp.write(f"{raw_file} {label_path} {' '.join(map(str, bin_label))}\n")

    train_gt_fp.close()


if __name__ == "__main__":
    # Podaj ścieżki do plików
    root = "C:\\Users\\Patry\\Desktop\\FinalProject\\data\\dataset\\Town03"
    json_file = os.path.join(root, "small_train_labels_10%.json")
    output_seg_dir = os.path.join(root, "segmentation_labels")
    train_gt_file = os.path.join(root, "train_gt.txt")

    # Przetwarzaj plik JSON
    process_json_file(root, json_file, output_seg_dir, train_gt_file)
