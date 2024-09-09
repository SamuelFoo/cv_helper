from pathlib import Path

import cv2
import numpy as np
import rosbag2_py
from cv_bridge import CvBridge
from rclpy.serialization import deserialize_message
from rosidl_runtime_py.utilities import get_message


def read_messages(bag_path: Path):
    reader = rosbag2_py.SequentialReader()
    reader.open(
        rosbag2_py.StorageOptions(uri=str(bag_path), storage_id="mcap"),
        rosbag2_py.ConverterOptions(
            input_serialization_format="cdr", output_serialization_format="cdr"
        ),
    )

    topic_types = reader.get_all_topics_and_types()

    def typename(topic_name):
        for topic_type in topic_types:
            if topic_type.name == topic_name:
                return topic_type.type
        raise ValueError(f"topic {topic_name} not in bag")

    while reader.has_next():
        topic, data, timestamp = reader.read_next()
        msg_type = get_message(typename(topic))
        msg = deserialize_message(data, msg_type)
        yield topic, msg, timestamp
    del reader


bridge = CvBridge()


def convert_ros_bag_mcap(bag_path: Path, vid_save_path: Path, img_topic: str) -> None:
    def get_next_image(msg):
        typename = type(msg).__name__
        if typename == "Image":
            img: np.ndarray = bridge.imgmsg_to_cv2(msg, "bgr8")
        elif typename == "CompressedImage":
            img: np.ndarray = bridge.compressed_imgmsg_to_cv2(msg, "bgr8")
        else:
            return None
        return img

    out = None

    for topic, msg, timestamp in read_messages(bag_path):
        if topic == img_topic:
            img = get_next_image(msg)

            if img is not None:
                if out is None:
                    img_height, img_width, _ = img.shape

                    # Initialize videowriter to output video
                    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
                    out = cv2.VideoWriter(
                        str(vid_save_path), fourcc, 24.0, (img_width, img_height)
                    )

                # Write to video
                out.write(img)

    if out is not None:
        out.release()
