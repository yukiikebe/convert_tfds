from src.make_tfds import CustomDataset
import os

def create_numerical_directory(base_path):
    if not os.path.exists(base_path):
        os.makedirs(base_path)

    existing_dirs = [d for d in os.listdir(base_path) if d.isdigit() and os.path.isdir(os.path.join(base_path, d))]
    if existing_dirs:
        new_dir_number = max(map(int, existing_dirs)) + 1
    else:
        new_dir_number = 0

    new_dir_path = os.path.join(base_path, str(new_dir_number))
    os.makedirs(new_dir_path)

    return new_dir_path

def main():
    # current_directory = os.getcwd()
    # data_directory = os.path.join(current_directory, 'data')
    # if not os.path.exists(data_directory):
    #     os.makedirs(data_directory)

    # new_data_directory = create_numerical_directory(data_directory)
    dataset_builder = CustomDataset(read_dir='1719814550')
    dataset_builder.download_and_prepare()
    # dataset = dataset_builder.as_dataset(split='train')
    
    # example_count = 0
    # for example in dataset.take(3):  # Inspect the first 5 examples
    #     print(example)
    #     example_count += 1

    # # Count the rest of the examples
    # for example in dataset.skip(3):
    #     example_count += 1

if __name__ == '__main__':
    main()