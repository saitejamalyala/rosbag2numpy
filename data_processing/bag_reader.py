from bagpy import bagreader
import os
from typing import Dict, List


class Read_Ros_Bag:
    """Class to Read RosBag file using bagreader from bagpy and extract messages and sequences"""

    def __init__(self, path) -> None:
        """Constructor for Read_Ros_Bag file
        Args:
            path ([os.path like]): path pointing towards location of rosbag file

        """
        self.path = path
        self.bag_name = os.path.split(self.path)[-1].split(".")[0]
        try:
            self.bag = bagreader(self.path).reader
        except:
            print("Bag reading Error")
        pass

    def msgs(self) -> Dict[str, List]:
        """For extracting msgs from Rosbag reader

        Returns:
            Dict[str, List]: list of messages and sequences of each kind
        """
        topic_list = [
            "/em/fused_grid",
            "/sensorik/axis_front/image/compressed",
            "/vehicle/odometry",
            "/function/pathing_mlteleop",
        ]
        pathing_msgs = []
        grid_msgs = []
        grid_seqs = []
        odom_msgs = []
        odom_seqs = []
        image_msgs = []
        image_seqs = []
        for topic, msg, t in self.bag.read_messages(topics=topic_list):

            if topic == "/function/pathing_mlteleop":
                pathing_msgs.append(msg)

            if topic == "/em/fused_grid":
                grid_msgs.append(msg)
                grid_seqs.append(msg.header.seq)

            if topic == "/vehicle/odometry":
                odom_msgs.append(msg)
                odom_seqs.append(msg.header.seq)

            if topic == "/sensorik/axis_front/image/compressed":
                image_msgs.append(msg)
                image_seqs.append(msg.header.seq)

        self.bag.close()

        data_params = {
            "pathing_msgs": pathing_msgs,
            "grid_msgs": grid_msgs,
            "grid_seqs": grid_seqs,
            "image_msgs": image_msgs,
            "image_seqs": image_seqs,
            "odom_msgs": odom_msgs,
            "odom_seqs": odom_seqs,
        }

        pathing_msgs = []
        grid_msgs = []
        grid_seqs = []
        odom_msgs = []
        odom_seqs = []
        image_msgs = []
        image_seqs = []

        return data_params
