from __future__ import print_function

import os
import time

import h5py
import malis
import numpy as np

from config import using_in_memory, simple_augmenting, body_ids_to_include
from config import mask_threshold, mask_dilation_steps

## Training datasets
train_dataset = []

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
        train_dataset.append(dataset)


for dataset in train_dataset:
    dataset['nhood'] = malis.mknhood3d()
    dataset['mask_dilation_steps'] = mask_dilation_steps
    dataset['simple_augment'] = simple_augmenting
    dataset['transform'] = {}
    dataset['transform']['scale'] = (0.9, 1.1)
    dataset['transform']['shift'] = (-0.1, 0.1)
    dataset['body_ids_to_include'] = body_ids_to_include

def convert_hdf5_to_in_memory(dataset, simple_augment=False):
    assert type(dataset['data']) is h5py.Dataset, type(dataset['data'])
    spatial_shape = dataset['data'].shape[-3:]
    data_file = dataset['data'].file
    dataset['data'] = np.array(dataset['data'], dtype=np.float32)
    data_file.close()
    dataset['data'] = dataset['data'].reshape(spatial_shape)
    dataset['data'] *= dataset.get('image_scaling_factor', 1)
    components_file = dataset['components'].file
    dataset['components'] = np.array(dataset['components'])
    components_file.close()
    dataset['components'] = dataset['components'].reshape(spatial_shape)
    dataset['label'] = malis.seg_to_affgraph(dataset['components'], dataset['nhood'])
    return dataset


if using_in_memory:
    print('converting to in memory')
    train_dataset = map(convert_hdf5_to_in_memory, train_dataset)
    if simple_augmenting:
        print('running simple augmentation')
        import PyGreentea as pygt
        train_dataset = pygt.augment_data_simple(train_dataset)
    for dataset in train_dataset:
        dataset['data'] = dataset['data'][None, :]
        dataset['components'] = dataset['components'][None, :]
        print(dataset['name'],
              'data shape:', str(dataset['data'].shape),
              'data mean: ', np.mean(dataset['data']),
              'components shape:', str(dataset['components'].shape))
        time.sleep(1)


print('Training set contains',
      len(train_dataset),
      'volumes:',
      [dataset['name'] for dataset in train_dataset],
      "with dtype/shapes",
      [(array.dtype, array.shape) for array in [dataset[key] for key in ("data", "components")] for dataset in train_dataset])

## Testing datasets
test_dataset = []
