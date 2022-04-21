# Installation

This document contains detailed instructions for installing the necessary dependencies for PyTracking. The instrustions have been tested on an Ubuntu 18.04 system. We recommend using the [install script](install.sh) if you have not already tried that.  

### Requirements       @ Installation/
* Conda installation with Python 3.7. If not already installed, install from https://www.anaconda.com/distribution/.
* Nvidia GPU.

## Step-by-step_instructions       @ Installation
#### Create_and_activate_a_conda_environment       @ Step-by-step_instructions/Installation
```bash
conda create --name pytracking python=3.7
conda activate pytracking
```

#### Install_PyTorch       @ Step-by-step_instructions/Installation
Install PyTorch with cuda10.  
```bash
conda install pytorch torchvision cudatoolkit=10.0 -c pytorch
```

**Note:**  
- It is possible to use any PyTorch supported version of CUDA (not necessarily v10).   
- For more details about PyTorch installation, see https://pytorch.org/get-started/previous-versions/.  

#### Install_matplotlib,_pandas,_tqdm,_opencv,_scikit-image,_visdom,_tikzplotlib,_gdown,_and_tensorboad       @ Step-by-step_instructions/Installation
```bash
conda install matplotlib pandas tqdm
pip install opencv-python visdom tb-nightly scikit-image tikzplotlib gdown
```


#### Install_the_coco_toolkit       @ Step-by-step_instructions/Installation
If you want to use COCO dataset for training, install the coco python toolkit. You additionally need to install cython to compile the coco toolkit.
```bash
conda install cython
pip install pycocotools
```


#### Install_ninja-build_for_Precise_ROI_pooling       @ Step-by-step_instructions/Installation
To compile the Precise ROI pooling module (https://github.com/vacancy/PreciseRoIPooling), you may additionally have to install ninja-build.
```bash
sudo apt-get install ninja-build
```
In case of issues, we refer to https://github.com/vacancy/PreciseRoIPooling. 

#### Install_Precise_ROI_pooling       @ Step-by-step_instructions/Installation
    @ Step-by-step_instructions/Installation
linking doesn't seem to work

```
ln -s trackers/pytracking/ltr/external/PreciseRoIPooling/src/prroi_pooling_gpu_impl.cu trackers/pytracking/ltr/external/PreciseRoIPoolingpytorch/prroi_pool/src/prroi_pooling_gpu_impl.cu
ln -s trackers/pytracking/ltr/external/PreciseRoIPooling/src/prroi_pooling_gpu_impl.cuh trackers/pytracking/ltr/external/PreciseRoIPoolingpytorch/prroi_pool/src/prroi_pooling_gpu_impl.cuh
```

so better copy
```
cp trackers/pytracking/ltr/external/PreciseRoIPooling/src/prroi_pooling_gpu_impl.cu trackers/pytracking/ltr/external/PreciseRoIPooling/pytorch/prroi_pool/src/prroi_pooling_gpu_impl.cu
cp trackers/pytracking/ltr/external/PreciseRoIPooling/src/prroi_pooling_gpu_impl.cuh trackers/pytracking/ltr/external/PreciseRoIPooling/pytorch/prroi_pool/src/prroi_pooling_gpu_impl.cuh
```

#### Install_jpeg4py       @ Step-by-step_instructions/Installation
In order to use [jpeg4py](https://github.com/ajkxyz/jpeg4py) for loading the images instead of OpenCV's imread(), install jpeg4py in the following way,  
```bash
sudo apt-get install libturbojpeg
pip install jpeg4py 
```

**Note:** The first step (```sudo apt-get install libturbojpeg```) can be optionally ignored, in which case OpenCV's imread() will be used to read the images. However the second step is a must.  

In case of issues, we refer to https://github.com/ajkxyz/jpeg4py.  


#### Setup_the_environment       @ Step-by-step_instructions/Installation
Create the default environment setting files. 
```bash
# Environment settings for pytracking. Saved at pytracking/evaluation/local.py
python -c "from pytracking.evaluation.environment import create_default_local_file; create_default_local_file()"

# Environment settings for ltr. Saved at ltr/admin/local.py
python -c "from ltr.admin.environment import create_default_local_file; create_default_local_file()"
```

You can modify these files to set the paths to datasets, results paths etc.  


#### Download_the_pre-trained_networks       @ Step-by-step_instructions/Installation
You can download the pre-trained networks from the [google drive folder](https://drive.google.com/drive/folders/1WVhJqvdu-_JG1U-V0IqfxTUa1SBPnL0O). 
The networks shoud be saved in the directory set by "network_path" in "pytracking/evaluation/local.py". By default, it is set to 
pytracking/networks.
You can also download the networks using the [gdown](https://github.com/wkentaro/gdown) python package.

```bash
# Download the default network for DiMP-50 and DiMP-18
gdown https://drive.google.com/uc\?id\=1qgachgqks2UGjKx-GdO1qylBDdB1f9KN -O pytracking/networks/dimp50.pth
gdown https://drive.google.com/uc\?id\=1MAjrRJDCbL0DSjUKFyDkUuYS1-cYBNjk -O pytracking/networks/dimp18.pth

# Download the default network for ATOM
gdown https://drive.google.com/uc\?id\=1VNyr-Ds0khjM0zaq6lU-xfY74-iWxBvU -O pytracking/networks/atom_default.pth

# Download the default network for ECO
gdown https://drive.google.com/uc\?id\=1aWC4waLv_te-BULoy0k-n_zS-ONms21S -O pytracking/networks/resnet18_vggmconv1.pth
```
