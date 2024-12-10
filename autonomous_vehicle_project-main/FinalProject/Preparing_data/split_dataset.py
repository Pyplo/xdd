import os
import json
import random

def split_dataset(input_file, output_dir, train_ratio=0.8):
    """
    Splits the dataset into training and validation sets based on a specified ratio.

    Args:
        input_file (str): Path to the JSON file containing the full dataset.
        output_dir (str): Directory where the split files will be saved.
        train_ratio (float): Proportion of the data to include in the training set (default is 0.8).
    """
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)

    # Load the dataset
    with open(input_file, 'r') as file:
        lines = file.readlines()

    # Shuffle the dataset to ensure random split
    random.shuffle(lines)

    # Calculate split index
    split_index = int(len(lines) * train_ratio)

    # Split into train and validate
    train_data = lines[:split_index]
    validate_data = lines[split_index:]

    # Save train set
    train_file = os.path.join(output_dir, 'train_labels.json')
    with open(train_file, 'w') as file:
        file.writelines(train_data)

    # Save validate set
    validate_file = os.path.join(output_dir, 'validate_labels.json')
    with open(validate_file, 'w') as file:
        file.writelines(validate_data)

    print(f"Dataset split completed. Files saved in {output_dir}:")
    print(f" - Training set: {train_file} ({len(train_data)} samples)")
    print(f" - Validation set: {validate_file} ({len(validate_data)} samples)")

if __name__ == "__main__":
    # Path to the input JSON file
    input_file = "C:/Users/Patry/Desktop/FinalProject/data/dataset/Town03/small_train_labels_10%.json"

    # Output directory for the split datasets
    output_dir = "C:/Users/Patry/Desktop/FinalProject/data/dataset/Town03/splited data"

    # Proportion of data for training (e.g., 80%)
    train_ratio = 0.8

    # Run the split
    split_dataset(input_file, output_dir, train_ratio)
