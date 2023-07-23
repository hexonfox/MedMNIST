# MedMNIST
Solving MedMNIST with ResNet-20, VGG-16 and ConvNet

### Create env and download dataset 
```python
# Create conda env
conda create -y -n medmnist python=3.9 && conda activate medmnist

# Install tensorflow, cuda and cudnn
conda install -y -c conda-forge cudatoolkit=11.8.0
python3 -m pip install nvidia-cudnn-cu11==8.6.0.163 tensorflow==2.13.0
mkdir -p $CONDA_PREFIX/etc/conda/activate.d
echo 'CUDNN_PATH=$(dirname $(python -c "import nvidia.cudnn;print(nvidia.cudnn.__file__)"))' >> $CONDA_PREFIX/etc/conda/activate.d/env_vars.sh
echo 'export LD_LIBRARY_PATH=$CONDA_PREFIX/lib/:$CUDNN_PATH/lib:$LD_LIBRARY_PATH' >> $CONDA_PREFIX/etc/conda/activate.d/env_vars.sh
source $CONDA_PREFIX/etc/conda/activate.d/env_vars.sh

# For some ubuntu versions additional steps are required ...
conda install -y -c nvidia cuda-nvcc=11.3.58
printf 'export XLA_FLAGS=--xla_gpu_cuda_data_dir=$CONDA_PREFIX/lib/\n' >> $CONDA_PREFIX/etc/conda/activate.d/env_vars.sh
source $CONDA_PREFIX/etc/conda/activate.d/env_vars.sh
mkdir -p $CONDA_PREFIX/lib/nvvm/libdevice
cp $CONDA_PREFIX/lib/libdevice.10.bc $CONDA_PREFIX/lib/nvvm/libdevice/

# Additional packages
pip install pillow

# Download medmnist datasets
wget -O pathmnist.npz "https://zenodo.org/record/6496656/files/pathmnist.npz?download=1"
wget -O octmnist.npz "https://zenodo.org/record/6496656/files/octmnist.npz?download=1"
wget -O tissuemnist.npz "https://zenodo.org/record/6496656/files/tissuemnist.npz?download=1"
```

### Run
```bash
# train
# model options: resnet20 vgg16 convnet
# dataset options: pathmnist octmnist tissuemnist
# will try to run on gpu by default
# specify gpu index if there is a gpu preference 
python train.py --gpu 0 --model resnet20 --dataset pathmnist

# view training
tensorboard --logdir=logs

# print result summary after training all datasets and models
python print_scores.py
```