import os,sys

import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
file_path = os.path.dirname(os.path.abspath(__file__))
root_path = os.path.abspath(os.path.join(file_path, os.pardir))


sess = tf.Session(config=tf.ConfigProto(allow_soft_placement=True,log_device_placement=False))


