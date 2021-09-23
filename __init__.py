import yaml
import os
def load_config(config_path):
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    return config

#print(os.getcwd())
config = load_config('config.yaml')

print(config)
#sweep_config = load_config('sweep.yaml')