import yaml
import os
SEED=2021

def load_config(config_path):
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    return config

#print(os.getcwd())
config = load_config('config.yaml')

#print(config)
#sweep_config = load_config('sweep.yaml')