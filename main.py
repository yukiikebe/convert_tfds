from src.make_tfds import CustomDataset
import os
import tensorflow as tf
import tensorflow_datasets as tfds

def main():
    read_dirs = [f'data/{i}' for i in range(53) if i != 11 and i != 24]
    
    # Create the dataset builder
    builder = CustomDataset(read_dirs=read_dirs)
    
    # Get the raw generator
    raw_generator = builder._generate_examples(read_dirs)
    
    # Inspect the raw data
    for key, value in raw_generator:
        print(f"Key: {key}")
        print(f"Episode metadata: {value['episode_metadata']}")
        print(f"Number of steps: {len(value['steps'])}")
        print(f"Keys of first step: {value['steps'][0].keys()}")
        print(f"Shape of action in first step: {value['steps'][0]['action'].shape}")
        break  # Just look at the first episode
    
    # Now try to build and load the dataset
    builder.download_and_prepare()
    ds = builder.as_dataset(split="train")
    
    # Inspect the loaded dataset
    for episode in ds.take(1):
        print("\nLoaded dataset:")
        print(f"Episode metadata: {episode['episode_metadata']}")
        print(f"Type of steps: {type(episode['steps'])}")
        
        steps = episode['steps']
        print("\nSteps structure:")
        for key, value in steps.items():
            if isinstance(value, dict):
                print(f"{key}:")
                for sub_key, sub_value in value.items():
                    print(f"  {sub_key}: shape {tf.shape(sub_value)}, dtype {sub_value.dtype}")
            else:
                print(f"{key}: shape {tf.shape(value)}, dtype {value.dtype}")
        
        # Access data for the first step
        print("\nFirst step data:")
        print(f"Action: {steps['action'][0]}")
        print(f"Language instruction: {steps['language_instruction'][0]}")
        print(f"Gripper position: {steps['observation']['gripper_position'][0]}")
        
        # Print the number of steps
        num_steps = tf.shape(steps['action'])[0]
        print(f"\nNumber of steps: {num_steps}")
        
        break

if __name__ == '__main__':
    main()