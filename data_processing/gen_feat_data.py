"""
# To be used after extracting the data from rosbag to compressed numpy arrays

# sample inputs required to run this script: path to folder inside scenario 
with _grid.npz, _init_path.npz (real world co-ordinates), _opt_path.npz(real world co-ordinates), _odo.npz(real world co-ordinates).

# output: single npz file consisting of costmap, feature vector, difference vector (init_path - opt_path), init_path, opt_path, car odo 
"""

from typing import List
from numpy import ndarray
import numpy as np
from fnmatch import fnmatch
import os
from pathlib import Path
from tqdm import trange

class FeatGen:
    """
    To generate feature vector and save it in a compressed .npz format 
    """
    def __init__(self,x_off:int,y_off:int,root_path) -> None:
        
        self.root_path = root_path
        self.file_pattern = '*_grid.npz'
        self.x_offset = x_off
        self.y_offset = y_off
        pass

    def get_all_grid_paths(self,print_path:bool=False)->List:
        """Get all .npz files with "_grid.npz" pattern in the specified root directory

        Args:
            self.root_path : Root path where all your _grid.npz are located
            self.file_pattern : pattern string to search
            print_path (bool, optional): If true prints paths of all files found. Defaults to False.

        Returns:
            List: list containing all paths where _grid.npz (file pattern string) files are located.

        """
        list_gridmap_loc=[]
        for path, subdirs, files in os.walk(self.root_path):
            for name in files:
                if fnmatch(name, self.file_pattern):
                    if print_path:
                        print(os.path.join(path, name))
                    list_gridmap_loc.append(os.path.join(path,name))

        return list_gridmap_loc

    def transform_coord(self,path,grid_info):
        assert grid_info.shape[0] == 3
        if len(path.shape)==2:
            assert path.shape[1]==2
            org = grid_info[0:2]
            res = grid_info[2]
            return (path-org)/res
        elif len(path.shape)==1:
            assert path.shape[0]==2
            org = grid_info[0:2]
            res = grid_info[2]
            return (path-org)/res

    def load_np_data(self,grid_path:str,print_shapes:bool=True):
        """load all numpy arrays (grid_data{categorical: 0,127}, _init_path, _opt_path, car_odo {all in real world coordinates})

        Args:
            grid_path (str): path to _grid.npz file

        Returns:
            multiple nd arrays: grid_data, grid_org_res, init_data, opt_data, odo_data
        """
        #0,127
        test_grid_data = np.load(file=grid_path)['grid_data']
        test_org_res = np.load(file=grid_path)['grid_org_res']
        # real world coords
        test_init_data = np.load(file=grid_path.replace('_grid','_init_path'))['init_path']
        test_opt_data = np.load(file=grid_path.replace('_grid','_opt_path'))['opt_path']
        test_odo_data = np.load(file=grid_path.replace('_grid','_odo'))['odo_data']

        if print_shapes:
            print(f"grid shape:{test_grid_data.shape},org_res:{test_org_res.shape},init_path:{test_init_data.shape}")

        return test_grid_data,test_org_res,test_init_data,test_opt_data,test_odo_data

    def calc_direction_cost(self,car_coords:ndarray,endpoint_unit_normal:ndarray,vector:ndarray)->ndarray:
        """To calculate direction cost for each cell in occupancy gridmap w.r.t to car location and destination co-ordinate.
            Calculates direction weight for the costmap. 
            step 1: calculate direction vector for every co-ordinate 
            step 2: normalize the vector calculated in step 1  

        Args:
            car_coords (ndarray): Car Current location
            endpoint_unit_normal (ndarray): Unit normal vector between car location and ending point
            vector (ndarray): specific co-ordinate

        Returns:
            [ndarray]: array of direction cost for that vector
        """
        direction = car_coords-vector
        direction_unit = direction/np.linalg.norm(x=direction)

        direction_cost = np.dot(direction_unit,endpoint_unit_normal)

        return direction_cost

    def calc_costmap(self,gridmap:ndarray,gd_org_res:ndarray,car_odo:ndarray,initial_path:ndarray,include_dir_cost:bool=True,distance_mag:float=0.23)->ndarray:
        """Function to transfor grid map to cost map

        Args:
            gridmap (ndarray): Binary occupancy grid map (1536 * 1536)- data range : 0,1
            car_odo (ndarray): Car location x,y,theta - transformed cordinate values (in range of 0-1536)
            initial_path (ndarray): initial path proposed by trajectory planning module (in range of 0-1536)
            distance_mag (float, optional): lower it is larger cost considered for the occupied cells. Defaults to 0.23.

        Returns:
            ndarray: Cost map  array
        """
        
        car_coords = car_odo[0:2]
        end_coords = initial_path[-1]
        
        # Normalize coordinates to 0-1536 range
        #car_coords = (car_coords-gd_org_res[0:2])/gd_org_res[2]
        #end_coords = (end_coords-gd_org_res[0:2])/gd_org_res[2]

        # Initialize distance and direction costmap 
        dist_costmap = np.zeros(shape=(1536,1536))
        dir_costmap =  np.zeros(shape=(1536,1536))

        ################# calculate distance costmap ################
        
        # get occuppied cell indices
        (y,x) = np.where(gridmap==np.max(gridmap))
        # create coordinate pairs of location of occupied cells
        coords = np.array((x,y)).T

        # eucledian difference
        distance = np.square(coords-car_coords)
        distance = np.sum(distance, axis=1)
        distance = 1/(distance**distance_mag)
        dist_costmap[(y,x)]=distance

        ################# Direction cost map ######################
        
        if include_dir_cost:

            # end point direction
            direction_vector = car_coords-end_coords

            # unit vector between, car location and end point
            end_dir_unit = direction_vector/np.linalg.norm(x=direction_vector)

            # calculate dir cost for all occupied cell locations
            direction_cost= np.asarray([self.calc_direction_cost(car_coords=car_coords,endpoint_unit_normal=end_dir_unit,vector=coord) for coord in coords])

            # zero down all opposite directions to end point
            direction_cost[np.where(direction_cost<0)]=0.0

            dir_costmap[(y,x)] = direction_cost
        else:
            dir_costmap = np.ones(shape=(1536,1536))

        # pointwise multiply distance cost and direction cost
        costmap = dist_costmap*dir_costmap

        # normalize cost map between [0,1]
        costmap = costmap/np.max(costmap)
        #cp=r'C:\Users\Teja\Documents\_INFOTECH\Thesis\sample_Ros_bag\npzcostmaps'
        #np.savez_compressed(os.path.join(cp,file_details.decode("utf-8").split('/')[-1]),grid_data=costmap.astype(np.float16))
        return costmap.astype(float)

    def get_feature_vector(self,gridmap:ndarray,init_path:ndarray,offset:List[int]=[13,20])->ndarray:
        """Get feature vector from grodmap, initial path , offset(Tuple of x offset and y offset)

        Args:
            gridmap (ndarray): Costmap (Obtained after transformation from binary occupancy grid map)
            init_path (ndarray): [description]
            offset ([type]): [description]

        Returns:
            ndarray: Feature vector of shape (25,4*offset[0]*y_offset[1])
        """
        feature_vector = np.zeros(shape=(init_path.shape[0],4*offset[0]*offset[1]))
        # np.array(init_path.shape[0],4*offset[0]*offset[1])
        for i in range(init_path.shape[0]):
            x_l,y_l = (init_path[i].astype(int)) - offset
            x_h,y_h = (init_path[i].astype(int)) + offset
            cell = 0
            #print("lower:",(x_l,y_l))
            #print("higher:",(x_h,y_h))

            for x in range(x_l,x_h):
                for y in range(y_l,y_h):
                    if cell<feature_vector.shape[1]:
                        #gridmap[row][column]
                        feature_vector[i][cell] = gridmap[y][x]
                        cell+=1
        return feature_vector

    def check_imbalance(self,init_path_arr:ndarray,opt_path_arr:ndarray)->List:
        """To check at which indices in an array initial path and optimized path are different from each other

        Args:
            init_path_arr ([ndarray]): initial paths
            opt_path_arr ([ndarray]): optimized paths

        Returns:
            List: List of indices
        """
        count=0
        zero_paths = 0
        indices = []
        assert len(opt_path_arr)==len(init_path_arr)
        for i in range(len(init_path_arr)):
            if np.array_equal(init_path_arr[i],opt_path_arr[i]):
                count +=1
            if not np.array_equal(init_path_arr[i],opt_path_arr[i]):
                indices.append(i)
        return indices    
                
    def create_and_save_costmap_diff_featurevector(self,list_gridmap_loc:List):

        # looping though all npz files

        for i in trange(len(list_gridmap_loc)):

            #i=0
            target_location = os.path.split(list_gridmap_loc[i])[0]

            np_grid_map,np_org_res,np_init_data,np_opt_data,np_odo_data = self.load_np_data(grid_path=list_gridmap_loc[i]) 

            ### initialize lists to fold modified data
            cost_maps = []
            init_paths=[]
            opt_paths=[]
            car_odo_data=[]
            diff_paths = []
            list_features = []

            print(f"number of samples : {np_grid_map.shape[0]}")

            ################# get unequal path indices #############
            """ Shape changes for every file , information about indices of samples that have opt path different from init path"""
            unequal_indices = self.check_imbalance(np_init_data,np_opt_data)


            # looping through all samples in loaded numpy file
            for j in range(np_grid_map.shape[0]):

                # to transform from real world coordinates to grid map co-ordinates
                # grid coords(after tranformation)- (0-1536)
                tf_np_init_data = self.transform_coord(np_init_data[j],np_org_res[j])
                tf_np_opt_data = self.transform_coord(np_opt_data[j],np_org_res[j])
                tf_np_odo_data = self.transform_coord(np_odo_data[j][0:2],np_org_res[j])

                
                total_cost_map = self.calc_costmap(gridmap=np_grid_map[j]/127,gd_org_res=np_org_res[j],
                                        car_odo=tf_np_odo_data,initial_path=tf_np_init_data,
                                        include_dir_cost=True)

                #total_cost_map = np_grid_map[j]/127 #(For binary gm), also change file extension name below. 
                ###### Calculate feature vector from costmap and init_path ############
                fv = self.get_feature_vector(gridmap=total_cost_map,init_path = tf_np_init_data,offset= [self.x_offset,self.y_offset])

                assert fv.shape[0]==tf_np_init_data.shape[0]
                
                # append eventually, Very inefficient way *** !!! correct it later !!!
                if not np.allclose(tf_np_init_data,tf_np_opt_data):
                    cost_maps.append(total_cost_map)
                    init_paths.append(tf_np_init_data)
                    opt_paths.append(tf_np_opt_data)
                    diff_paths.append(tf_np_init_data-tf_np_opt_data)
                    car_odo_data.append(tf_np_odo_data)
                    list_features.append(fv)


            assert len(init_paths)==len(opt_paths)==len(diff_paths)==len(car_odo_data)==len(cost_maps)#np_grid_map.shape[0]
            
            print(f'post processing samples:{len(init_paths)}')
            ################ convert data to nd arrays to save in .npz #########################

            # costmap data range: 0.0-1.0 
            arr_np_cost_map = np.array(cost_maps)

            # all co-ordinates date range - [0.0-1536.0] , in grid co-ordinates
            arr_np_init_path = np.array(init_paths)
            arr_np_opt_path = np.array(opt_paths)
            arr_np_diff_path = np.array(diff_paths)
            arr_np_odo = np.array(car_odo_data)

            # list of features each of shape 25,4*x_offset*y_offset
            arr_np_fv = np.array(list_features)

            # list of indices- integers in list
            arr_neq_indices = np.array(unequal_indices)

            ############### save data in the same location as source path ##########################
            np.savez_compressed(
                list_gridmap_loc[i].replace("_grid","cm_fv_paths_odo_unequal"),
                costmap = arr_np_cost_map,
                init_path = arr_np_init_path,
                opt_path = arr_np_opt_path,
                diff_path = arr_np_diff_path,
                odo_data = arr_np_odo, 
                feature_vector = arr_np_fv,
                unequal_path_idx = arr_neq_indices,
                )


if __name__ == "__main__":

    source_path = r'C:\Users\Teja\Documents\_INFOTECH\Thesis\sample_Ros_bag\np_data\raw_data_wo_img'

    feat = FeatGen(x_off=13,y_off=20, root_path=source_path)

    #get all gridmap npz files
    grid_paths = feat.get_all_grid_paths(print_path=True)

    # load data, get all corresponding data in that respective folder for every gridmap npz
    #test_grid_data,test_org_res,test_init_data,test_opt_data,test_odo_data = feat.load_np_data(grid_path=grid_paths[0])

    # print to test shapes of loaded data
    #print(test_grid_data.shape,test_org_res.shape,test_init_data.shape,test_opt_data.shape,test_odo_data.shape)

    #check unequal indices functionality
    #print(f"Unequal Indices: {feat.check_imbalance(test_init_data,test_opt_data)}")

    #saves new npz files in the same location as source path
    feat.create_and_save_costmap_diff_featurevector(list_gridmap_loc=grid_paths)

