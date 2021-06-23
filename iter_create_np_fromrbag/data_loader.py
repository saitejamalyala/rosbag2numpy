import tensorflow as tf
import random
from glob import glob
from typing import List,Tuple
from pathlib import Path
from sklearn.model_selection import train_test_split
AUTOTUNE = tf.data.experimental.AUTOTUNE


class dataset_loader:

    def __init__(self) -> None:
        pass