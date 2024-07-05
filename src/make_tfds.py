import os
import tensorflow as tf
import tensorflow_datasets as tfds
import json
from PIL import Image
import numpy as np
from typing import List

os.environ["NO_GCE_CHECK"] = "true"

gpu_devices = tf.config.experimental.list_physical_devices('GPU')
for device in gpu_devices:
    tf.config.experimental.set_memory_growth(device, True)


class CustomDataset(tfds.core.GeneratorBasedBuilder):
    VERSION = tfds.core.Version('1.0.0')
    RELEASE_NOTES = {
      '1.0.0': 'Initial release.',
    }
    
    def __init__(self, read_dirs: List[str], **kwargs):
        super().__init__(**kwargs)
        self.read_dirs = read_dirs

    def _info(self):
        return tfds.core.DatasetInfo(
            builder=self,
            description="Custom dataset containing robot arm telemetry.",
            features=tfds.features.FeaturesDict({
                "episode_metadata": tfds.features.FeaturesDict({
                    "recording_folderpath": tfds.features.Text(),
                    "file_path0": tfds.features.Text(),
                    "file_path1": tfds.features.Text(),
                }),
                "steps": tfds.features.Sequence({
                    "language_instruction": tfds.features.Text(),
                    "observation": tfds.features.FeaturesDict({
                        "gripper_position": tfds.features.Tensor(shape=(1,), dtype=tf.int32),
                        "cartesian_position": tfds.features.Tensor(shape=(6,), dtype=tf.float64),
                        "joint_position": tfds.features.Tensor(shape=(6,), dtype=tf.float64),
                        "head_image_left": tfds.features.Image(shape=(180, 320, 3)),
                        "exterior_image_1_left": tfds.features.Image(shape=(180, 320, 3)),
                    }),
                    "action_dict": tfds.features.FeaturesDict({
                        "gripper_position": tfds.features.Tensor(shape=(1,), dtype=tf.float32),
                        "gripper_velocity": tfds.features.Tensor(shape=(1,), dtype=tf.float64),
                        "cartesian_position": tfds.features.Tensor(shape=(6,), dtype=tf.float64),
                        "cartesian_velocity": tfds.features.Tensor(shape=(6,), dtype=tf.float64),
                        "joint_position": tfds.features.Tensor(shape=(6,), dtype=tf.float64),
                        "joint_velocity": tfds.features.Tensor(shape=(6,), dtype=tf.float64),
                    }),
                    "action": tfds.features.Tensor(shape=(7,), dtype=tf.float64),
                }),
            }),
            supervised_keys=None,
        )

    def _split_generators(self, dl_manager):
        return [
            tfds.core.SplitGenerator(
                name=tfds.Split.TRAIN,
                gen_kwargs={'read_dirs': self.read_dirs},
            ),
        ]

    def _generate_examples(self, read_dirs):
        for episode_idx, episode in enumerate(read_dirs):
            print(f"Processing episode: {episode}")
            recording_dir_path0 = os.path.join(episode, "color_0")
            recording_dir_path1 = os.path.join(episode, "color_1")
            png_dir_path0 = os.path.join(episode, "color_0_png")
            png_dir_path1 = os.path.join(episode, "color_1_png")
            robot_data_path = os.path.join(episode, "png_robot_data.json")

            if os.path.isdir(recording_dir_path0) and os.path.isdir(recording_dir_path1):
                with open(robot_data_path, "r") as f:
                    data = json.load(f)
                steps = []
                for idx, step in enumerate(data):
                    try:
                        head_image_path = os.path.join(png_dir_path1, step["png_cam2"])
                        exterior_image_path = os.path.join(png_dir_path0, step["png_cam1"])

                        head_image = Image.open(head_image_path).resize((320, 180))
                        exterior_image = Image.open(exterior_image_path).resize((320, 180))

                        head_image = np.array(head_image)
                        exterior_image = np.array(exterior_image)

                        action = step["TargetQd"] + [step["Gripper_position"]]

                        step_data = {
                            "language_instruction": "Catch object and move and place it in the other spot",
                            "observation": {
                                "gripper_position": np.array([step["ActualGripper"]], dtype=np.int32),
                                "cartesian_position": np.array(step["ActualTCPPose"], dtype=np.float64),
                                "joint_position": np.array(step["ActualQ"], dtype=np.float64),
                                "head_image_left": head_image,
                                "exterior_image_1_left": exterior_image,
                            },
                            "action_dict": {
                                "gripper_position": np.array([step["Gripper_position"]/255], dtype=np.float32),
                                "gripper_velocity": np.array([step["Gripper_velocity"]], dtype=np.float64),
                                "cartesian_position": np.array(step["TargetTCPPose"], dtype=np.float64),
                                "cartesian_velocity": np.array(step["TargetTCPSpeed"], dtype=np.float64),
                                "joint_position": np.array(step["TargetQ"], dtype=np.float64),
                                "joint_velocity": np.array(step["TargetQd"], dtype=np.float64),
                            },
                            "action": np.array(action, dtype=np.float64),
                        }
                        steps.append(step_data)
                    except Exception as e:
                        print(f"Error processing step {idx} in directory {episode}: {e}")
                
                episode_data = {
                    "episode_metadata": {
                        "recording_folderpath": episode,
                        "file_path0": os.path.join(recording_dir_path0, os.listdir(recording_dir_path0)[0]),
                        "file_path1": os.path.join(recording_dir_path1, os.listdir(recording_dir_path1)[0]),
                    },
                    "steps": steps
                }
                
                yield f"{episode_idx}", episode_data

    # def _info(self):
    #     return tfds.core.DatasetInfo(
    #         builder=self,
    #         description=("Custom dataset containing robot arm telemetry."),
    #         features=tfds.features.FeaturesDict({
    #             "episode_metadata": tfds.features.FeaturesDict({
    #                 "recording_folderpath": tfds.features.Text(),
    #                 "file_path0": tfds.features.Text(),
    #                 "file_path1": tfds.features.Text(),
    #             }),
    #             "steps": tfds.features.Sequence({
    #                 "language_instruction": tfds.features.Text(),
    #                 "observation": {
    #                     "gripper_position": tfds.features.Tensor(shape=(1,), dtype=np.int32),
    #                     "cartesian_position": tfds.features.Tensor(shape=(6,), dtype=np.float64),
    #                     "joint_position": tfds.features.Tensor(shape=(6,), dtype=np.float64),
    #                     "head_image_left": tfds.features.Image(shape=(180, 320, 3)),
    #                     "exterior_image_1_left": tfds.features.Image(shape=(180, 320, 3)),
    #                 },
    #                 "action_dict": {
    #                     "gripper_position": tfds.features.Tensor(shape=(1,), dtype=np.float32),
    #                     "gripper_velocity": tfds.features.Tensor(shape=(1,), dtype=np.float64),
    #                     "cartesian_position": tfds.features.Tensor(shape=(6,), dtype=np.float64),
    #                     "cartesian_velocity": tfds.features.Tensor(shape=(6,), dtype=np.float64),
    #                     "joint_position": tfds.features.Tensor(shape=(6,), dtype=np.float64),
    #                     "joint_velocity": tfds.features.Tensor(shape=(6,), dtype=np.float64),
    #                 },
    #                 "action": tfds.features.Tensor(shape=(7,), dtype=np.float64),
    #             }),
    #         }),
    #         supervised_keys=None,
    #     )

    # def _split_generators(self, dl_manager):
    #     return {
    #         "train": self._generate_examples(self.read_dirs)
    #     }

    # def _generate_examples(self, read_dirs):
    #     for episode in read_dirs:
    #         print(episode)
    #         recording_dir_path0 = os.path.join(episode, "color_0")
    #         recording_dir_path1 = os.path.join(episode, "color_1")
    #         png_dir_path0 = os.path.join(episode, "color_0_png")
    #         png_dir_path1 = os.path.join(episode, "color_1_png")
    #         robot_data_path = os.path.join(episode, "png_robot_data.json")

    #         if os.path.isdir(recording_dir_path0) and os.path.isdir(recording_dir_path1):
    #             with open(robot_data_path, "r") as f:
    #                 data = json.load(f)
    #             steps = []
    #             for idx, step in enumerate(data):
    #                 try:
    #                     head_image_path = os.path.join(png_dir_path1, step["png_cam2"])
    #                     exterior_image_path = os.path.join(png_dir_path0, step["png_cam1"])

    #                     head_image = Image.open(head_image_path).resize((320, 180))
    #                     exterior_image = Image.open(exterior_image_path).resize((320, 180))

    #                     head_image = np.array(head_image)
    #                     exterior_image = np.array(exterior_image)

    #                     action = step["TargetQd"] + [step["Gripper_position"]]

    #                     steps.append({
    #                         "language_instruction": "Catch object and move and place it in the other spot",
    #                         "observation": {
    #                             "gripper_position": np.array([step["ActualGripper"]], dtype=np.int32),
    #                             "cartesian_position": np.array(step["ActualTCPPose"], dtype=np.float64),
    #                             "joint_position": np.array(step["ActualQ"], dtype=np.float64),
    #                             "head_image_left": head_image,
    #                             "exterior_image_1_left": exterior_image,
    #                         },
    #                         "action_dict": {
    #                             "gripper_position": np.array([step["Gripper_position"]/255], dtype=np.float32),
    #                             "gripper_velocity": np.array([step["Gripper_velocity"]], dtype=np.float64),
    #                             "cartesian_position": np.array(step["TargetTCPPose"], dtype=np.float64),
    #                             "cartesian_velocity": np.array(step["TargetTCPSpeed"], dtype=np.float64),
    #                             "joint_position": np.array(step["TargetQ"], dtype=np.float64),
    #                             "joint_velocity": np.array(step["TargetQd"], dtype=np.float64),
    #                         },
    #                         "action": np.array(action, dtype=np.float64),
    #                     })
    #                 except Exception as e:
    #                     print(f"Error processing step {idx} in directory {episode}: {e}")
                
    #             yield episode, {
    #                 "episode_metadata": {
    #                     "recording_folderpath": episode,
    #                     "file_path0": os.path.join(recording_dir_path0, os.listdir(recording_dir_path0)[0]),
    #                     "file_path1": os.path.join(recording_dir_path1, os.listdir(recording_dir_path1)[0]),
    #                 },
    #                 "steps": steps
    #             }