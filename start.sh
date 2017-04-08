#!/usr/bin/env bash
rm snapshots/*

sudo mount --make-shared /nrs/turaga

export NAME=$(basename "$PWD")

nvidia-docker rm $NAME

NV_GPU=4 \
    nvidia-docker run -d \
    -u `id -u $USER` \
    -v $(pwd):/workspace \
    -v /groups/turaga/home:/groups/turaga/home \
    -v /nrs/turaga:/nrs/turaga:shared \
    -v /groups/turaga/home/grisaitisw/src/blob_exclusion_PyGreentea:/opt/PyGreentea:ro \
    --name $NAME \
    turagalab/greentea:libdnn-caffe_gt-2016.12.05-pygt-0.9.4b \
    python -u train.py

#    -v $(pwd)/PyGreentea:/opt/PyGreentea:ro \

