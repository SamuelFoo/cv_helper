from pathlib import Path

import cv2
import numpy as np
from cv_bridge import CvBridge
from rosbags.highlevel import AnyReader
from rosbags.interfaces import Connection

bridge = CvBridge()
import argparse


def get_ros_bag_topics(bag_path: Path):
    with AnyReader([bag_path]) as reader:
        return reader.topics


def convert_ros_bag(bag_path: Path, vid_save_path: Path, img_topic: str) -> None:
    with AnyReader([bag_path]) as reader:

        def get_next_image(connection: Connection, rawdata: bytes):
            msg = reader.deserialize(rawdata, connection.msgtype)
            if connection.msgtype == "sensor_msgs/msg/Image":
                img: np.ndarray = bridge.imgmsg_to_cv2(msg, "bgr8")
            elif connection.msgtype == "sensor_msgs/msg/CompressedImage":
                img: np.ndarray = bridge.compressed_imgmsg_to_cv2(msg, "bgr8")
            else:
                return None
            return img

        out = None

        # Iterate over messages
        for connection, _, rawdata in reader.messages():
            if connection.topic == img_topic:

                # Get next image
                img: np.ndarray = get_next_image(connection=connection, rawdata=rawdata)

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

        out.release()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Extract and encode video from bag files."
    )
    parser.add_argument(
        "--infile",
        "-i",
        action="store",
        default=None,
        help="Destination of the ROS bag file.",
    )
    parser.add_argument(
        "--outfile",
        "-o",
        action="store",
        default=None,
        help="Destination of the video file.",
    )
    parser.add_argument(
        "--image-topic",
        "-t",
        action="store",
        default=None,
        help="Name of ROS topic that contains the images.",
    )

    args = parser.parse_args()

    infile = args.infile
    outfile = args.outfile
    img_topic = args.image_topic

    convert_ros_bag(
        bag_path=Path(infile),
        vid_save_path=Path(outfile),
        img_topics=img_topic,
    )
