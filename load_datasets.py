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

base_dir = '/groups/turaga/home/turagas/data/FlyEM/fibsem_medulla_7col'
for name in [
    'trvol-250-1-h5',
    'trvol-250-2-h5',
    'tstvol-520-1-h5'
    ]:
    dataset = dict()
    dataset['name'] = "FlyEM {}".format(name)
    image_file = h5py.File(os.path.join(base_dir, name, 'im_uint8.h5'), 'r')
    dataset['data'] = image_file['main']
    components_file = h5py.File(os.path.join(base_dir, name, 'groundtruth_seg_thick.h5'), 'r')
    dataset['components'] = components_file['main']
    dataset['image_scaling_factor'] = 1.0 / (2.0 ** 8)
    train_dataset.append(dataset)

base_dir = '/nobackup/turaga/grisaitisw/data/toufiq_mushroom'
for name in ['4400']:
    dataset = dict()
    dataset['name'] = "toufiq-mushroom-{}".format(name)
    image_file = h5py.File(os.path.join(base_dir, name, 'image_from_png_files.h5'), 'r')
    dataset['data'] = image_file['main']
    components_file = h5py.File(os.path.join(base_dir, name, 'components_eroded_by_1.h5'), 'r')
    dataset['components'] = components_file['stack']
    dataset['image_scaling_factor'] = 1.0
    train_dataset.append(dataset)

base_dir = '/nobackup/turaga/grisaitisw/data/pb/'
for name in ['pb']:
    dataset = dict()
    dataset['name'] = 'pb {}'.format(name)
    image_file = h5py.File(os.path.join(base_dir, name, 'image_from_png_files.h5'), 'r')
    dataset['data'] = image_file['main']
    components_file = h5py.File(os.path.join(base_dir, name, 'components_eroded_by_1.h5'), 'r')
    dataset['components'] = components_file['stack']
    dataset['image_scaling_factor'] = 1.0
    train_dataset.append(dataset)

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
train_dataset.extend([dataset] * 100)

dataset = dict()
dataset['name'] = 'dvid_mb6'
hostname = 'slowpoke3'
port = 32770
node = '6a5a7387b4ce4333aa18d9c8d8647f58'
dataset['data'] = DVIDDataInstance(hostname, port, node, 'grayscale')
dataset['components'] = DVIDDataInstance(hostname, port, node, 'alpha_123_labels')
dataset['minimum_component_size'] = minimum_component_size
dataset['component_erosion_steps'] = component_erosion_steps
dataset['image_scaling_factor'] = 1.0 / (2.0 ** 8)
dataset['mask_threshold'] = mask_threshold
train_dataset.extend([dataset] * 100)

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
