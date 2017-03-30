import numpy as np

DEBUG = False
SAVE_IMAGES = False

# hyperparameters
base_learning_rate = 1e-4
fmap_start = 24
net_input_shape = [132] * 3
net_output_shape = [44] * 3
use_deconvolution_uppath = False  # must be True if input and output shapes are the same
malis_split_component_phases = False

# data prep paramaters
mask_threshold = 0
mask_dilation_steps = 0
using_in_memory = False
simple_augmenting = True
minimum_component_size = 0
body_names_to_exclude = []
component_erosion_steps = 0
random_state = np.random.RandomState(seed=0)
from bodies_tstvol_520_1_h5 import body_list
body_ids_to_include = random_state.choice(body_list, int(0.8 * len(body_list)))
print(len(body_ids_to_include))

# runtime settings
training_gpu_device = 0
testing_gpu_device = training_gpu_device
image_saving_frequency = 500
