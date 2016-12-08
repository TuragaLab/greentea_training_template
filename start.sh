rm snapshots/*

sudo mount --make-shared /nrs/turaga

export NAME=$(basename "$PWD")

nvidia-docker rm $NAME

NV_GPU=1 \
    nvidia-docker run -d \
    -u `id -u $USER` \
    -v $(pwd):/workspace \
    -v /groups/turaga/home:/groups/turaga/home \
    -v /nrs/turaga:/nrs/turaga:shared \
    --name $NAME \
    turagalab/greentea:cuda8.0-cudnn5-caffe_gt-2016.12.02-pygt-0.9.4b \
    python -u train.py

