from __future__ import print_function

import os
import time

import h5py
import malis
import numpy as np

from dvision import DVIDDataInstance
import PyGreentea as pygt

from config import using_in_memory, simple_augmenting
from config import mask_threshold, mask_dilation_steps
from config import minimum_component_size, body_names_to_exclude
from config import component_erosion_steps

## Training datasets
train_dataset = []

dataset = dict()
dataset['name'] = 'dvid_fib25'
hostname = 'slowpoke3'
port = 32773
node = 'e402c09ddd0f45e980d9be6e9fcb9bd0'
dataset['data'] = DVIDDataInstance(hostname, port, node, 'grayscale')
dataset['components'] = DVIDDataInstance(hostname, port, node, 'labels1104')
dataset['body_names_to_exclude'] = body_names_to_exclude
dataset['component_erosion_steps'] = component_erosion_steps
dataset['image_scaling_factor'] = 1.0 / (2.0 ** 8)
dataset['mask_threshold'] = mask_threshold
train_dataset.append(dataset)

for dataset in train_dataset:
    dataset['nhood'] = malis.mknhood3d()
    dataset['mask_dilation_steps'] = mask_dilation_steps
    dataset['transform'] = {}
    dataset['transform']['scale'] = (0.9, 1.1)
    dataset['transform']['shift'] = (-0.1, 0.1)

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
      [dataset['name'] for dataset in train_dataset])

## Testing datasets
test_dataset = []
