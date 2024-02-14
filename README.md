# Visual Foundational Models Are Clustering Learners

## Installation

Below are quick steps for installation:

```shell
conda create -n open-mmlab python=3.8 pytorch=1.10 cudatoolkit=11.3 torchvision==0.11.0 -c pytorch -y
conda activate open-mmlab
pip3 install openmim
mim install mmcv-full
cd clustnet
pip3 install -e .
```

Please refer to [install.md](https://mmclassification.readthedocs.io/en/latest/install.html) for more detailed installation and dataset preparation.

## Training
We followed the common usage of the mmclassification and check [mmclassification](https://github.com/open-mmlab/mmpretrain) for more training information.

In particular, we use the slurm system to train our model. [Slurm](https://slurm.schedmd.com/) is a good job scheduling system for computing clusters.

On a cluster managed by Slurm, you can use `slurm_train.sh` to spawn training jobs. It supports both single-node and multi-node training.

The basic usage is as follows.

```shell
OMP_NUM_THREADS=1 [GPUS=${GPUS}] ./tools/slurm_train.sh ${PARTITION} ${JOB_NAME} ${CONFIG_FILE} ${WORK_DIR}
```

When using Slurm, the port option need to be set in one of the following ways:

1. Set the port through `--options`. This is more recommended since it does not change the original configs.

   ```shell
   OMP_NUM_THREADS=1 CUDA_VISIBLE_DEVICES=0,1,2,3 GPUS=4 ./tools/slurm_train.sh ${PARTITION} ${JOB_NAME} config1.py ${WORK_DIR} --options 'dist_params.port=29500'
   OMP_NUM_THREADS=1 CUDA_VISIBLE_DEVICES=4,5,6,7 GPUS=4 ./tools/slurm_train.sh ${PARTITION} ${JOB_NAME} config2.py ${WORK_DIR} --options 'dist_params.port=29501'
   ```

2. Modify the config files to set different communication ports.

   In `config1.py`, set

   ```python
   dist_params = dict(backend='nccl', port=29500)
   ```

   In `config2.py`, set

   ```python
   dist_params = dict(backend='nccl', port=29501)
   ```

   Then you can launch two jobs with `config1.py` and `config2.py`.

   ```shell
   OMP_NUM_THREADS=1 CUDA_VISIBLE_DEVICES=0,1,2,3 GPUS=4 ./tools/slurm_train.sh ${PARTITION} ${JOB_NAME} config1.py ${WORK_DIR}
   OMP_NUM_THREADS=1 CUDA_VISIBLE_DEVICES=4,5,6,7 GPUS=4 ./tools/slurm_train.sh ${PARTITION} ${JOB_NAME} config2.py ${WORK_DIR}
   ```

Note that:
- The configs are made for 8-GPU training. To train on another number of GPUs, change the `GPUS`.
- If you want to measure the inference time, please change the number of gpu to 1 for inference.
- We set `OMP_NUM_THREADS=1` by default, which achieves the best speed on our machines, please change it as needed.
