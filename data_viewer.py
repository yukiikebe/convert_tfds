import tensorflow_datasets as tfds
import tensorflow as tf
import numpy as np
from PIL import Image
import cv2
import time

def as_gif(images, path="temp.gif"):
  # Render the images as the gif (15Hz control frequency):
  images[0].save(path, save_all=True, append_images=images[1:], duration=int(1000/15), loop=0)
  gif_bytes = open(path,"rb").read()
  return gif_bytes


ds = tfds.load("custom_dataset", data_dir="tensorflow_datasets", split="train")


def process_episode(episode):
    steps = episode['steps']
    num_steps = tf.shape(steps['action'])[0]
    
    for i in range(num_steps):
        # Access data for step i
        action = steps['action'][i]
        language_instruction = steps['language_instruction'][i]
        gripper_position = steps['observation']['gripper_position'][i]
        head_image = steps['observation']['head_image_left'][i]
        exterior_image = steps['observation']['exterior_image_1_left'][i]
        
        # Process the step data as needed
        # For example, you might want to use the images and action in a model:
        # predicted_action = your_model(head_image, exterior_image, language_instruction)
        # loss = compute_loss(predicted_action, action)
        
        # For now, let's just print some information
        print(f"Step {i}:")
        print(f"  Action: {action}")
        print(f"  Language instruction: {language_instruction.numpy().decode('utf-8')}")
        print(f"  Gripper position: {gripper_position}")
        print(f"  Head image shape: {tf.shape(head_image)}")
        print(f"  Exterior image shape: {tf.shape(exterior_image)}")
        
        if i >= 2:  # Just print the first 3 steps as an example
            break

# In your main function:
for episode in ds.take(1):
    process_episode(episode)

raise

print(len(ds))

images = []
for episode in ds.take(1):
    print(episode["episode_metadata"])
    print(episode['steps'].keys())

    print(len(episode["steps"]["action"]))
    print(len(episode["steps"]["action_dict"]["gripper_position"]))
    print(len(episode["steps"]["language_instruction"]))
    print(len(episode["steps"]["observation"]))
    for i, step in enumerate(episode["steps"]):
        print(type(step), step)
        print(episode.keys())
        print(step.keys())
        raise
        images.append(
            np.concatenate((
                step["observation"]["exterior_image_1_left"].numpy(),
                step["observation"]["head_image_left"].numpy(),
                # step["observation"]["wrist_image_left"].numpy(),
            ), axis=1)
        )
        print(step["action"])
        print(step['language_instruction'])
        # print(step["action_dict"]["cartesian_position"].numpy(), step["action_dict"]["gripper_position"].numpy())
        # print(images[-1][:,:,::-1].shape)
        # print(step["observation"]['cartesian_position'].numpy(), step["observation"]['gripper_position'].numpy())
        cv2.imshow('Video Stream', images[-1][:,:,::-1]) #obs["agentview_image"])

        # Wait for 1 ms for a key event and check if the key 'q' is pressed
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

        time.sleep(0.1)

cv2.destroyAllWindows()
