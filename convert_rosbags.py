from pathlib import Path

import cv2
import numpy as np
from rosbags.rosbag1 import Reader


def convert_ros1_bag(
    bag_path: Path,
    vid_save_path: Path,
    img_height: int,
    img_width: int,
    img_topic: str,
    convert_rgb: bool = False,
):
    """Convert ROS 1 Bags to MP4.

    Args:
        bag_path (Path): path to bag file
        vid_save_path (Path): path to video file
        img_height (int): image height
        img_width (int): image width
        convert_rgb (bool, optional): If True, will convert from BGR to RGB. Defaults to False.
    """

    img_height = 720
    img_width = 1280

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    out = cv2.VideoWriter(str(vid_save_path), fourcc, 24.0, (img_width, img_height))

    with Reader(bag_path) as reader:
        # for connection in reader.connections:
        #     print(connection.topic, connection.msgtype)

        # Iterate over messages
        for connection, _, rawdata in reader.messages():
            if connection.topic == img_topic:
                # Convert to NumPy array
                img: np.ndarray = np.fromstring(rawdata, dtype=np.uint8)

                # Crop out extra pixels
                num_pixels = 3 * img_height * img_width
                img: np.ndarray = img[:num_pixels]
                img: np.ndarray = img.reshape(img_height, img_width, 3)

                # Optionally convert from BGR to RGB
                if convert_rgb:
                    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

                # Write to video
                out.write(img)

        out.release()
