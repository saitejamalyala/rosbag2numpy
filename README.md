# rosbag2numpy
Convert data in rosbag format to numpy.
Specific to ros message structure in rosbag.

The repository works for the following [ros message structure](https://github.com/saitejamalyala/rosbag2numpy/tree/main/media)

1. [Extract Data from ros bag and save it in compressed numpy](https://github.com/saitejamalyala/rosbag2numpy/blob/main/iter_create_np_fromrbag/iter_all_rosbags.py) Format (Helps in moving and manipulating extracted data)
  * Change source directory (Root directory where rosbags are located) [here](https://github.com/saitejamalyala/rosbag2numpy/blob/2e9235302835792c5cfda9693c2fa88cdbd34f80/iter_create_np_fromrbag/iter_all_rosbags.py#L23)
  * Change target directory(To store compressed numpy arrays) [here] (https://github.com/saitejamalyala/rosbag2numpy/blob/2e9235302835792c5cfda9693c2fa88cdbd34f80/iter_create_np_fromrbag/iter_all_rosbags.py#L25)

2. Convert to Tf records to build efficient tensorflow Data pipelines
