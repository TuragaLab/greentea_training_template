rm -r snapshots
mkdir snapshots
rm net_*
rm loss*
rm *pyc

# Mesos task info
MESOS_TASK_NAME="task-$(date +%s)"

# Container info
DOCKER_IMAGE="turagalab/greentea:cuda8.0-cudnn6_libdnn-caffe_gt-2017.04.17-pygt-0.9.4b"
N_CPU_SHARES="5"
MEMORY_MB="25600"
N_GPUS="1"

echo "find stdout & stderr on the node executing your container at:"
echo "find \$(find /var/lib/mesos/slaves -name $MESOS_TASK_NAME -type d) -name std*"

mesos-execute \
    --master=$(mesos-resolve `cat /etc/mesos/zk`) \
    --name=$MESOS_TASK_NAME \
    --docker_image=$DOCKER_IMAGE \
    --framework_capabilities="GPU_RESOURCES" \
    --resources="cpus:$N_CPU_SHARES;mem:$MEMORY_MB;gpus:$N_GPUS" \
    --command="cd $(pwd) && PYTHONPATH=$(pwd)/PyGreentea:\$PYTHONPATH && python -u train.py" \
    --env="{\"HOME\": \"$HOME\"}" \
    --containerizer=mesos \
    --volumes=file://$(pwd)/volumes.json

    # --command='PATH=~/anaconda2/bin:$PATH && env && which python' \
