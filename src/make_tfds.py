import tensorflow as tf
import tensorflow_datasets as tfds
import json
import os
from PIL import Image
import numpy as np
import base64
from io import BytesIO

class CustomDataset(tfds.core.GeneratorBasedBuilder):
    """Example of a custom dataset loader."""
    VERSION = tfds.core.Version('1.0.0')

    def __init__(self, read_dir, **kwargs):
        self.read_dir = read_dir
        super().__init__(**kwargs)

    def _info(self):
        return tfds.core.DatasetInfo(
            builder=self,
            description=("Custom dataset containing robot arm telemetry."),
            features=tfds.features.FeaturesDict({
                "episode_metadata": tfds.features.FeaturesDict({
                    "recording_folderpath": tfds.features.Text(),
                    "file_path0": tfds.features.Text(),
                    "file_path1": tfds.features.Text(),
                }),
                "steps": tfds.features.Sequence({
                    "language_instruction": tfds.features.Text(),
                    "observation": tfds.features.FeaturesDict({
                        "gripper_position": tfds.features.Tensor(shape=(1,), dtype=np.int32),
                        "cartesian_position": tfds.features.Tensor(shape=(6,), dtype=np.float64),
                        "joint_position": tfds.features.Tensor(shape=(6,), dtype=np.float64),
                        "head_image_left": tfds.features.Image(shape=(180, 320, 3)),
                        "exterior_image_1_left": tfds.features.Image(shape=(180, 320, 3)),
                    }),
                    "action_dict": tfds.features.FeaturesDict({
                        "gripper_position": tfds.features.Tensor(shape=(1,), dtype=np.int32),
                        "gripper_velocity": tfds.features.Tensor(shape=(1,), dtype=np.float64),
                        "cartesian_position": tfds.features.Tensor(shape=(6,), dtype=np.float64),
                        "cartesian_velocity": tfds.features.Tensor(shape=(6,), dtype=np.float64),
                        "joint_position": tfds.features.Tensor(shape=(6,), dtype=np.float64),
                        "joint_velocity": tfds.features.Tensor(shape=(6,), dtype=np.float64),
                    }),
                    # "discount": tfds.features.Tensor(shape=(), dtype=np.float32),
                    # "reward": tfds.features.Tensor(shape=(), dtype=np.float32),
                    "action": tfds.features.Tensor(shape=(7,), dtype=np.float64),
                }),
            }),
            supervised_keys=None,
        )
    
    def _split_generators(self, dl_manager):
        """Returns SplitGenerators."""
        # Assumes data is in one single JSON file

        robot_data_path = f"{self.read_dir}/png_robot_data.json"
        return [
            tfds.core.SplitGenerator(
                name=tfds.Split.TRAIN,
                gen_kwargs={"data_path": robot_data_path},
            ),
        ]

    def _generate_examples(self, data_path):
        """Yields examples."""
        # for i in range(50):
        #     read_dir = i
        recording_dir_path0 = f"{self.read_dir}/color_0"
        recording_dir_path1 = f"{self.read_dir}/color_1"
        png_dir_path0 = f"{self.read_dir}/color_0_png"
        png_dir_path1 = f"{self.read_dir}/color_1_png"

        robot_data_path = f"{self.read_dir}/png_robot_data.json"
        if os.path.isdir(recording_dir_path0):
            files0 = os.listdir(recording_dir_path0)
        if os.path.isdir(recording_dir_path1):
            files1 = os.listdir(recording_dir_path1)

        with open(robot_data_path, "r") as f:
            data = json.load(f)
            for idx, step in enumerate(data):
                try:
                    head_image_path = os.path.join(png_dir_path1, step["png_cam2"])
                    exterior_image_path = os.path.join(png_dir_path0, step["png_cam1"])

                    head_image = Image.open(head_image_path)
                    exterior_image = Image.open(exterior_image_path)

                    # Resize images to the required shape (180, 320)
                    head_image = head_image.resize((320, 180))
                    exterior_image = exterior_image.resize((320, 180))

                    # Convert images to numpy arrays
                    head_image = np.array(head_image)
                    exterior_image = np.array(exterior_image)

                    action = step["TargetQd"] + [step["Gripper_position"]]

                    yield idx, {
                        "episode_metadata": {
                            "recording_folderpath": os.path.join(self.read_dir),
                            "file_path0": os.path.join(recording_dir_path0, files0[0]),
                            "file_path1": os.path.join(recording_dir_path1, files1[0]),
                        },
                        "steps": [
                            {
                                "language_instruction": "Catch object and move and place it in the other spot",
                                "observation": {
                                    "gripper_position": [step["ActualGripper"]],
                                    "cartesian_position": step["ActualTCPPose"],
                                    "joint_position": step["ActualQ"],
                                    "head_image_left": head_image,
                                    "exterior_image_1_left": exterior_image,
                                },
                                "action_dict": {
                                    "gripper_position": [step["Gripper_position"]],
                                    "gripper_velocity": [step["Gripper_velocity"]],
                                    "cartesian_position": step["TargetTCPPose"],
                                    "cartesian_velocity": step["TargetTCPSpeed"],
                                    "joint_position": step["TargetQ"],
                                    "joint_velocity": step["TargetQd"],
                                },
                                # "discount": step["discount"],
                                # "reward": step["reward"],
                                "action": action,
                            } 
                        ]
                    }
                except Exception as e:
                    print(f"Error processing step {idx}: {e}")
        print("Finished generating examples.")