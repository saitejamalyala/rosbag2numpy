import matplotlib.pyplot as plt


def plot_scene(features,normalized_coords:bool=True,normalized_factor = 1):    
    grid_map = features["grid_map"]
    grid_org = features["grid_org_res"] #[x,y,resolution]
    left_bnd = features["left_bnd"]
    right_bnd = features["right_bnd"]
    init_path = features["init_path"]
    opt_path = features["opt_path"]
    car_odo = features["car_odo"]

    predict_path = features["predictions"]
    file_details=features["file_details"]

    #print(type(grid_map))
    
    plt.figure(figsize=(10, 10))
    #ax=fig.add_subplot(1,1,1)
    if normalized_coords:
        #res = grid_org[2]
            plt.plot(
                left_bnd[:, 0]*normalized_factor,
                left_bnd[:, 1]*normalized_factor,
                "-.",
                color="magenta",
                markersize=0.5,
                linewidth=0.5,
            )

            plt.plot(
                init_path[:, 0]*normalized_factor,
                init_path[:, 1]*normalized_factor,
                "o-",
                color="lawngreen",
                markersize=1,
                linewidth=1,
            )
            plt.plot(
                opt_path[:, 0]*normalized_factor,
                opt_path[:, 1]*normalized_factor,
                "--",
                color="yellow",
                markersize=1,
                linewidth=1,
            )
            
            plt.plot(
                predict_path[:, 0]*normalized_factor,
                predict_path[:, 1]*normalized_factor,
                "--",
                color="orange",
                markersize=1,
                linewidth=1,
            )

            plt.plot(
                right_bnd[:, 0]*normalized_factor,
                right_bnd[:, 1]*normalized_factor,
                "-.",
                color="magenta",
                markersize=0.5,
                linewidth=0.5,
            )

            plt.plot(
                car_odo[0],
                car_odo[1],
                "r*",
                color="red",
                markersize=8,
            )
 


    else:
        res = grid_org[2]
        plt.plot((left_bnd[:,0]-grid_org[0])/res,(left_bnd[:,1]-grid_org[1])/res,'-.', color='magenta',markersize=0.5, linewidth=0.5)

        plt.plot((init_path[:,0]-grid_org[0])/res,(init_path[:,1]-grid_org[1])/res,'o-', color='lawngreen',markersize=1, linewidth=1)

        plt.plot((opt_path[:,0]-grid_org[0])/res,(opt_path[:,1]-grid_org[1])/res,'--', color='yellow',markersize=1, linewidth=1)

        plt.plot((predict_path[:,0]-grid_org[0])/res,(predict_path[:,1]-grid_org[1])/res,'--', color='orange',markersize=1, linewidth=1)

        plt.plot((right_bnd[:,0]-grid_org[0])/res,(right_bnd[:,1]-grid_org[1])/res, '-.',color='magenta',markersize=0.5, linewidth=0.5)

        plt.plot((car_odo[0]-grid_org[0])/res,(car_odo[1]-grid_org[1])/res,'r*', color = 'red',markersize=8)
        #print((car_odo[0]-grid_org[0])/res,(car_odo[1]-grid_org[1])/res)

    plt.legend(['Left bound', 'gt_init_path', 'gt_opt_path','predicted_path','right bound', 'car_centre'], loc='lower left')

    plt.imshow(grid_map.astype(float),origin="lower")

    plt.title(f"{file_details}\nTest Index: {features['testidx']}")
    #save_fig_dir = '/netpool/work/gpu-3/users/malyalasa/New_folder/rosbag2numpy/test_results/all_testset_results'

    """ 
        root_dir = '/netpool/work/gpu-3/users/malyalasa/New_folder/rosbag2numpy/test_results/after_normalization'
        model_name = model_path.split('/')[-3]
        scene_dir = os.path.split(file_details.numpy().decode("utf-8"))[0]
        scene_dir = os.path.split(scene_dir)[0]
        save_dir=os.path.join(root_dir,model_name,scene_dir)
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)
        plt.savefig(f"{save_dir}/Test_index_{features['testidx']}.png",dpi=200)
    """
    
    plt.show()
    #plt.close()
    
