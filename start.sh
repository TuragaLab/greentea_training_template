rm snapshots/*

sudo mount --make-shared /nobackup/turaga

export NAME=$(basename "$PWD")

nvidia-docker rm $NAME

NV_GPU=0 \
    nvidia-docker run -d \
    -u `id -u $USER` \
    -v $(pwd):/workspace \
    -v /groups/turaga/home:/groups/turaga/home \
    -v /nobackup/turaga:/nobackup/turaga:shared \
    --name $NAME \
    turagalab/greentea:cudnn5-caffe_gt-pygt-0.9 \
    python -u train.py

    # -v $(pwd)/PyGreentea:/opt/PyGreentea \
