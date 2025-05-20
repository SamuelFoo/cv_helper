# ROS Bags Conversion

`convert_rosbags.py` is a replacement for `convert_rosbags_old.py` as the former handles both ROS 1 and ROS 2 bags using the same API. Both depend on `rosbags`, which can be installed using [pip](https://pypi.org/project/rosbags/) (does not require ROS).

Neither work for `.mcap` bags, so use `convert_rosbags_mcap.py` instead. `convert_rosbags_mcap.py` depends on `rosbag2_py`, which is a [ROS package](https://index.ros.org/p/rosbag2_py/).
