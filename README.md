# MedMNIST
Solving MedMNIST with ResNet-20, VGG-16 and ConvNet

### Create env and download dataset 
```python
# Create conda env
conda create -y -n medmnist python=3.7 && conda activate medmnist

# Install tensorflow, cuda and cudnn
conda install -y tensorflow-gpu=1.15.0 keras=2.3.1 h5py=2.8.0

# Additional packages
pip install pillow pandas

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

# train all combinations 
./train.sh

# view training
tensorboard --logdir=logs

# print result summary after training all datasets and models
python print_scores.py
```