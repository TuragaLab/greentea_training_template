DEBUG = False
SAVE_IMAGES = False

# hyperparameters
base_learning_rate = 5e-5
fmap_start = 40
net_input_shape = [100] * 3
net_output_shape = [100] * 3
use_deconvolution_uppath = True  # must be True if input and output shapes are the same
malis_split_component_phases = True

# data prep paramaters
mask_threshold = 0.5
mask_dilation_steps = 1
using_in_memory = False
simple_augmenting = False
minimum_component_size = 1000
body_names_to_exclude = ['glia']
component_erosion_steps = 1

# runtime settings
training_gpu_device = 0
testing_gpu_device = training_gpu_device
image_saving_frequency = 10
