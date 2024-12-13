import os


def create_smaller_dataset_file(dataset, keep: int = 10):
    """
    Creates a smaller dataset file based on the percentage of frames to keep.

    Args:
        dataset: Absolute path to a train_labels.json file
        keep: Percentage of frames to keep (e.g., 20 for 20%)

    Returns:
        None: Creates a smaller dataset file in the same directory.
    """
    base_path = os.path.dirname(dataset)
    output_dataset = f"small_train_labels_{keep}%.json"

    # Check if output file already exists
    if os.path.isfile(os.path.join(base_path, output_dataset)):
        raise Exception(f'Output file "{output_dataset}" already exists')

    # Read the source dataset
    with open(dataset, 'r') as file:
        source_data = file.readlines()

    # Calculate step size and select frames
    step_size = max(1, int(100 / keep))
    keep_data = source_data[::step_size]

    # Write the reduced dataset to the output file
    with open(os.path.join(base_path, output_dataset), 'w') as file:
        file.writelines(keep_data)

    print(f"Smaller dataset created: {output_dataset}")


if __name__ == '__main__':
    # Example usage: Create a smaller dataset keeping 20% of the original frames
    create_smaller_dataset_file(
        'C:\\Users\\Patry\\Desktop\\FinalProject\\data\\dataset\\Town03\\train_gt.json',
        keep=10
    )
