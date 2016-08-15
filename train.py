from __future__ import print_function

import sys

import numpy as np

# Load PyGreentea
# Relative path to where PyGreentea resides
pygt_path = '../../PyGreentea'
sys.path.append(pygt_path)
import PyGreentea as pygt

from config import DEBUG, SAVE_IMAGES
pygt.DEBUG = DEBUG
pygt.SAVE_IMAGES = SAVE_IMAGES

from load_datasets import train_dataset, test_dataset
from config import base_learning_rate, training_gpu_device, testing_gpu_device
from config import image_saving_frequency
from config import malis_split_component_phases

import mknet

# Set train options
class TrainOptions:
    loss_function = "euclid"
    loss_output_file = "log/loss.log"
    test_output_file = "log/test.log"
    test_interval = 2000
    scale_error = True
    training_method = "affinity"
    recompute_affinity = True
    train_device = training_gpu_device
    test_device = testing_gpu_device
    # test_net='net_test.prototxt'
    test_net=None
    max_iter = 10000
    snapshot = 10000
    loss_snapshot = 2000
    snapshot_prefix = 'net'
    save_image_snapshot_period = image_saving_frequency

options = TrainOptions()

# Set solver options
print('Initializing solver...')
solver_config = pygt.caffe.SolverParameter()
solver_config.train_net = 'net_train_euclid.prototxt'

solver_config.type = 'Adam'
solver_config.base_lr = base_learning_rate
solver_config.momentum = 0.99
solver_config.momentum2 = 0.999
solver_config.delta = 1e-8
solver_config.weight_decay = 0.000005

solver_config.lr_policy = 'inv'
solver_config.gamma = 0.0001
solver_config.power = 0.75

solver_config.max_iter = options.max_iter
solver_config.snapshot = options.snapshot
solver_config.snapshot_prefix = options.snapshot_prefix
solver_config.display = 1

# Set devices
print('Setting devices...')
pygt.caffe.enumerate_devices(False)
pygt.caffe.set_devices(tuple(set((options.train_device, options.test_device))))

# First training method
solverstates = pygt.getSolverStates(solver_config.snapshot_prefix);
if (len(solverstates) == 0 or solverstates[-1][0] < solver_config.max_iter):
    solver, test_net = pygt.init_solver(solver_config, options)
    if (len(solverstates) > 0):
        solver.restore(solverstates[-1][1])
    pygt.train(solver, test_net, train_dataset, test_dataset, options)

# Second training method
solverstates = pygt.getSolverStates(solver_config.snapshot_prefix);
if (solverstates[-1][0] >= solver_config.max_iter):
    # Modify some solver options
    solver_config.max_iter = 400000
    solver_config.train_net = 'net_train_malis.prototxt'
    options.loss_function = 'malis'
    options.malis_split_component_phases = malis_split_component_phases
    # Initialize and restore solver
    solver, test_net = pygt.init_solver(solver_config, options)
    if (len(solverstates) > 0):
        solver.restore(solverstates[-1][1])
    pygt.train(solver, test_net, train_dataset, test_dataset, options)
