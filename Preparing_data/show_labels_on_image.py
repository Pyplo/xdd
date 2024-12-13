import json
import os

import cv2


def get_lane_color(i):
    lane_colors = [(255, 0, 0), (0, 255, 0), (0, 0, 255), (255, 255, 0), (255, 0, 255)]
    return lane_colors[i % 5]


def input_images(input_file, data_root):
    """ This method provides an easy way to visually validate train data by drawing labels on the frames and displaying them

    Args:
        input_file: labels file (filepath relative to data_root)
        data_root: path to dataset
    """
    with open(os.path.join(data_root, input_file)) as file:
        lines = file.readlines()

    for line in lines:
        dict = json.loads(line)

        image_path = os.path.join(data_root, dict['raw_file'])
        if not os.path.isfile(image_path):
            print(f"File not found: {image_path}")
            continue

        image = cv2.imread(image_path)
        if image is None:
            print(f"Could not read image: {image_path}")
            continue

        for i in range(len(dict['lanes'])):
            lane = dict['lanes'][i]
            for j in range(len(dict['h_samples'])):
                if lane[j] != -2:
                    cv2.circle(image, (lane[j], dict['h_samples'][j]), 5, get_lane_color(i), -1)

        print(f"Displaying: {dict['raw_file']}")
        cv2.imshow('video', image)
        cv2.waitKey(1000)


if __name__ == '__main__':
    input_images('data/dataset/Town03/small_train_labels_10%.json', 'C:/Users/Patry/Desktop/FinalProject/')

