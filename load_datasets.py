from __future__ import print_function

import os

import h5py
import malis

from dvision import DVIDDataInstance
from config import simple_augmenting
from config import mask_threshold, mask_dilation_steps
from config import minimum_component_size, dvid_body_names_to_exclude
from config import component_erosion_steps


train_datasets = []

base_dir = '/nrs/turaga/data/FlyEM/fibsem_medulla_7col'
for name in ['tstvol-520-1']:
    image_file = h5py.File(os.path.join(base_dir, name, 'im_uint8.h5'), 'r')
    components_file = h5py.File(os.path.join(base_dir, name, 'groundtruth_seg_thick.h5'), 'r')
    mask_file = h5py.File(os.path.join(base_dir, name, 'mask.h5'), 'r')
    for h5_key in (
        "tstvol-520-1-h5_y0_x0_xy0_angle000.0",
        "tstvol-520-1-h5_y0_x0_xy0_angle022.5",
        "tstvol-520-1-h5_y0_x0_xy0_angle045.0",
        "tstvol-520-1-h5_y0_x0_xy0_angle067.5",
        "tstvol-520-1-h5_y0_x0_xy0_angle090.0",
        "tstvol-520-1-h5_y0_x0_xy0_angle112.5",
        "tstvol-520-1-h5_y0_x0_xy0_angle135.0",
        "tstvol-520-1-h5_y0_x0_xy0_angle157.5",
        ):
        print("Adding", h5_key)
        dataset = dict()
        dataset['name'] = "FlyEM {0} {1}".format(name, h5_key)
        dataset['data'] = image_file[h5_key]
        dataset['components'] = components_file[h5_key]
        dataset['mask'] = mask_file[h5_key]
        dataset['image_scaling_factor'] = 1.0 / (2.0 ** 8)
        train_datasets.append(dataset)

# fib25 = dict(
#     name="FIB-25 train",
#     data=DVIDDataInstance("slowpoke3", 32788, "213", "grayscale"),
#     components=DVIDDataInstance("slowpoke3", 32788, "213", "groundtruth_pruned"),
#     image_scaling_factor=1.0 / (2.0 ** 8),
#     component_erosion_steps=component_erosion_steps,
#     bounding_box=((2000, 5006), (1000, 5000), (2000, 6000)),  # train region
#     dvid_body_names_to_exclude=dvid_body_names_to_exclude,
# )
# train_datasets.extend([fib25] * 8)

for dataset in train_datasets:
    dataset['nhood'] = malis.mknhood3d().astype(int)
    dataset['mask_threshold'] = mask_threshold
    dataset['mask_dilation_steps'] = mask_dilation_steps
    dataset['minimum_component_size'] = minimum_component_size
    dataset['simple_augment'] = simple_augmenting
    dataset['transform'] = {}
    dataset['transform']['scale'] = (0.9, 1.1)
    dataset['transform']['shift'] = (-0.1, 0.1)


print('Training set contains',
      len(train_datasets),
      'volumes:',
      [dataset['name'] for dataset in train_datasets],
      "with dtype/shapes",
      [(array.dtype, array.shape) for array in [dataset[key] for key in ("data", "components")] for dataset in train_datasets])

## Testing datasets
test_datasets = []
