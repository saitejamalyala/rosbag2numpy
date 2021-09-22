import yaml
import os
def load_config(config_path):
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    return config

#print(os.getcwd())
config = load_config('/netpool/work/gpu-3/users/malyalasa/New_folder/rosbag2numpy/config.yaml')

#print(config)
#sweep_config = load_config('sweep.yaml')