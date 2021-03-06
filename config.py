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
mask_threshold = 0.5
mask_dilation_steps = 0
using_in_memory = False
simple_augmenting = True
minimum_component_size = 0
dvid_body_names_to_exclude = []
component_erosion_steps = 1

# runtime settings
training_gpu_device = 0
testing_gpu_device = training_gpu_device
image_saving_frequency = 500
