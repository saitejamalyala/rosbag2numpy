import PIL.Image as Image
import numpy as np
from numpy import ndarray
import numpy
from tqdm import tqdm
import sys
import io
from typing import Dict, List, Any, Optional, Tuple

'''
sys.path.insert(
    0, "~/ddad/application/urban/remote_services/teleop_machine_learning/src/training"
)'''


class np_maker():
    """
    Description of np_maker

    Attributes:
        pathing_msgs (type):
        grid_msgs (type):
        grid_seqs (type):
        image_msgs (type):
        image_seqs (type):
        odom_msgs (type):
        odom_seqs (type):
        max_length (type):
        extract_msg_cnt (type):
        _ZERO_PADDING (type):

    Args:
        max_length=25 (undefined):
        ZERO_PADDING=False (undefined):
        **params (undefined):

    """

    def __init__(
        self,
        max_length: int = 25,
        ZERO_PADDING: bool = False,
        **params: Dict[str, List]
    ):
        """create ndarrays from msgs extracted from rosbags
        Args:
            max_length (int, optional): Maximum length of path(intial and optimal). Defaults to 25.
            ZERO_PADDING (bool, optional): if the path is to be padded with zeros. Defaults to False.
        """
        self.pathing_msgs = params.get("pathing_msgs")
        self.grid_msgs = params.get("grid_msgs")
        self.grid_seqs = params.get("grid_seqs")

        self.image_msgs = params.get("image_msgs")
        self.image_seqs = params.get("image_seqs")

        self.odom_msgs = params.get("odom_msgs")
        self.odom_seqs = params.get("odom_seqs")

        self.max_length = max_length
        self.extract_msg_cnt = 220

        self._ZERO_PADDING = ZERO_PADDING

    @staticmethod
    def __np_reshape_frm_list(list_path,new_shape):
        return np.reshape((np.asarray(list_path)),new_shape)
    
    @staticmethod
    def __np_hstack_list(self,list1,list2,new_shape):
        assert len(list1)==len(list2)
        np_list1 = self.__np_reshape_frm_list(list1,new_shape)
        np_list2 = self.__np_reshape_frm_list(list2,new_shape)
        return np.hstack((np_list1,np_list2))
   
    @staticmethod
    def __padd_values(n: int, val: float) -> List[float]:
        listofvalues = [val] * n
        return listofvalues

    @staticmethod
    def __padded_path(self, unpadded_path):
        padded_path = unpadded_path.extend(
            self.__padd_values(self.max_length - len(unpadded_path), 0)
        )
        return padded_path

    @staticmethod
    def __construct_path(list_of_points: List) -> Tuple[List[float], List[float]]:
        """construct path from points in form of a list

        Args:
            list_of_points (List): List of objects which have point x and y stored

        Returns:
            Tuple[List[float],List[float]]: path_x(list of float) and path_y(list of float)
        """
        path_x = []
        path_y = []
        for path in list_of_points:
            path_x.append(path.point.x)
            path_y.append(path.point.y)

        return path_x, path_y

    @staticmethod
    def __assert_all_lengths(
        self, path1_x: List, path1_y: List, path2_x: List, path2_y: List
    ):
        # check if every path is as long as the maximum length
        assert len(path1_x) == self.max_length
        assert len(path1_y) == self.max_length
        assert len(path2_x) == self.max_length
        assert len(path2_y) == self.max_length

    def create_np_path(self) -> Tuple[ndarray, ndarray]:
        """To create ndarray version of initial path and optimal path data

        Returns:
            Tuple[ndarray,ndarray]: initial_path and opt_path
        """

        list_all_init_path = []
        list_all_opt_path = []
        print("Converting path data to numpy........ ")

        # Enumerate through pathing messages and construct paths
        for i, examplary_pathing_msg in enumerate(tqdm(self.pathing_msgs)):

            # print("examplary_pathing_msg")

            init_path_x, init_path_y = self.__construct_path(
                examplary_pathing_msg.path_options[0].reference_path
            )

            opt_path_x, opt_path_y = self.__construct_path(
                examplary_pathing_msg.path_options[2].reference_path
            )

            # padding paths to fixed length of size self.max_length

            if self._ZERO_PADDING:
                if len(init_path_x) < self.max_length:
                    init_path_x.extend(
                        self.__padd_values(self.max_length - len(init_path_x), 0)
                    )
                    # init_path_x = self.__padded_path(self,init_path_x)
                    init_path_y.extend(
                        self.__padd_values(self.max_length - len(init_path_y), 0)
                    )

                if len(opt_path_x) < self.max_length:
                    opt_path_x.extend(
                        self.__padd_values(self.max_length - len(opt_path_x), 0)
                    )
                    opt_path_y.extend(
                        self.__padd_values(self.max_length - len(opt_path_y), 0)
                    )

            # padding with last ordinate of the path
            else:
                if len(init_path_x) < self.max_length:
                    init_path_x.extend(
                        self.__padd_values(
                            self.max_length - len(init_path_x), init_path_x[-1]
                        )
                    )
                    # init_path_x = self.__padded_path(self,init_path_x)
                    init_path_y.extend(
                        self.__padd_values(
                            self.max_length - len(init_path_y), init_path_y[-1]
                        )
                    )

                if len(opt_path_x) < self.max_length:
                    opt_path_x.extend(
                        self.__padd_values(
                            self.max_length - len(opt_path_x), opt_path_x[-1]
                        )
                    )
                    opt_path_y.extend(
                        self.__padd_values(
                            self.max_length - len(opt_path_y), opt_path_y[-1]
                        )
                    )

            # check for lengths
            self.__assert_all_lengths(
                self, init_path_x, init_path_y, opt_path_x, opt_path_y
            )

            # pair x and y ordinates to make it a co-ordinate
            np_init_path = self.__np_hstack_list(
                self, init_path_x, init_path_y, new_shape=(self.max_length, 1))
            np_opt_path = self.__np_hstack_list(
                self, opt_path_x, opt_path_y, new_shape=(self.max_length, 1))

            # append paths from all pathing messages
            list_all_init_path.append(np_init_path)
            list_all_opt_path.append(np_opt_path)

        # Convert the all paths appended list to numpy array
        np_all_initp = np.asarray(list_all_init_path)
        np_all_optp = np.asarray(list_all_opt_path)

        return np_all_initp, np_all_optp

    def create_np_grid(self) -> ndarray:
        """To create ndarray version of grid data

        Returns:
            ndarray: ndarray of grid data(shape:1526*1526*1)
        """
        list_all_grid_data = []
        print("Converting grid data to numpy........ ")
        for i, examplary_pathing_msg in enumerate(tqdm(self.pathing_msgs)):

            # get sequences aligning with pathing messages
            grid_seq = (
                examplary_pathing_msg.path_options[1]
                .reference_path[0]
                .left_boundaries.lane
            )

            """
            grid is saved as vector in grid_msg.data
            grid_msg.info gives information about properties, i.e. width, height, resolution, position in odometry frame
            """
            grid_idx = self.grid_seqs.index(grid_seq)
            grid_msg = self.grid_msgs[grid_idx]

            grid_data = np.asarray(grid_msg.data)
            grid_data = grid_data.reshape(grid_msg.info.width, grid_msg.info.height)

            # append grid data
            list_all_grid_data.append(grid_data)

        # converted appended grids to np array
        np_all_grid_data = np.asarray(list_all_grid_data)

        return np_all_grid_data

    def create_np_img(self) -> ndarray:
        """To create ndarray version of image data(all images)

        Returns:
            ndarray: array of images
        """
        list_all_img_data = []
        print("Converting image data to numpy........ ")
        for i, examplary_pathing_msg in enumerate(tqdm(self.pathing_msgs)):

            image_seq = (
                examplary_pathing_msg.path_options[1]
                .reference_path[0]
                .left_boundaries.road
            )
            image_idx = self.image_seqs.index(image_seq)
            image_msg = self.image_msgs[image_idx]

            # Convert compressed image(byte string) to RAW using
            img = Image.open(io.BytesIO(image_msg.data))
            # convert PIL image object to numpy array
            np_img = np.asarray(img)
            list_all_img_data.append(np_img)

        np_all_img_data = np.asarray(list_all_img_data)

        return np_all_img_data

    def create_np_odo(self) -> ndarray:
        """To create ndarray version of odo data

        Returns:
            ndarray: array of odometry data (position)
        """
        list_all_odo_data = []
        print("Converting Odometer data to numpy........ ")
        for examplary_pathing_msg in enumerate(tqdm(self.pathing_msgs)):

            odom_seq = (
                examplary_pathing_msg.path_options[1]
                .reference_path[0]
                .left_boundaries.obstacle
            )
            odom_idx = self.odom_seqs.index(odom_seq)
            odom_msg = self.odom_msgs[odom_idx]

            # adding only position (x,y), check odo_msg for other data(heading, angle and others) thats available
            list_all_odo_data.append((odom_msg.pos_x, odom_msg.pos_y))

        np_all_odo_data = np.asfarray(list_all_odo_data)

        return np_all_odo_data


"""
    # functionality check
    convert_2_np = np_maker(pathing_msgs,_MAX_LENGTH)

    np_init_path,np_opt_path = convert_2_np.create_np_path()

    np_grid_data = convert_2_np.create_np_grid()

    np_image_data = convert_2_np.create_np_img()


    print((np_init_path[0]))
    print(np.shape(np_opt_path))

    print(np.shape(np_grid_data))

    print(np.shape(np_image_data))

    from matplotlib import pyplot as plt
    #plt.matshow(np_grid_data[0])

    plt.plot(np_init_path[0,:15,0],np_init_path[0,:15,1])
    plt.show()
"""
