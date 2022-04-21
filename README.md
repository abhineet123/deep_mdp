<!-- No Heading Fix -->

Welcome to the home of **Deep MDP**  - a fast, modular and parallelized Python implementation of the [MDP framework](https://github.com/yuxng/MDP_Tracking) for Multi-Object Tracking with added support for deep learning. 

<!-- MarkdownTOC -->

- [Python dependencies](#python_dependencies_)
- [Installation in Linux](#installation_in_linu_x_)
    - [imbalanced-dataset-sampler](#imbalanced_dataset_sample_r_)
    - [libsvm](#libsv_m_)
    - [C_modules](#c_modules_)
- [Installation in Windows](#installation_in_window_s_)

<!-- /MarkdownTOC -->

<a id="python_dependencies_"></a>
# Python dependencies

```
pip3 install xlwt
```

<a id="installation_in_linu_x_"></a>
# Installation in Linux

<a id="imbalanced_dataset_sample_r_"></a>
## imbalanced-dataset-sampler 

```
cd models/thirdparty/imbalanced-dataset-sampler && python3 setup.py install

```

<a id="libsv_m_"></a>
## libsvm

 * install [libsvm](https://www.csie.ntu.edu.tw/~cjlin/libsvm/):
```
cd models/thirdparty/libsvm-3.23/python

make

cp ../libsvm.so.2 /usr/local/lib
mkdir ../../../libsvm
mkdir ../../../libsvm/linux
cp ../libsvm.so.2 ../../../libsvm/linux

mkdir ../../../libsvm
mkdir ../../../libsvm/linux
cp ../libsvm.so.2 ../../../libsvm/linux/libsvm.so.2

cd -

```

<a id="c_modules_"></a>
## C_modules

```
cd cmodules && mkdir build && cd build
 
cmake ..
 
sudo make install
```


<a id="installation_in_window_s_"></a>
# Installation in Windows

 * install [libsvm](https://www.csie.ntu.edu.tw/~cjlin/libsvm/) by following the instructions [here](https://stackoverflow.com/a/32358275) and [here](https://stackoverflow.com/a/12877497)

 * install [cmake](https://cmake.org/)

* install Visual Studio 2015 by following instructions [here](https://stackoverflow.com/a/44290942)

* install [OpenCV](https://opencv.org/) by following the instructions [here](https://docs.opencv.org/3.2.0/d3/d52/tutorial_windows_install.html)

* installation using pre built binaries is not guaranteed to work so build from source if cmake does not find OpenCV in the next step

* create an environment variable called `OpenCV_DIR` pointing to the OpenCV installation or build folder containing the `OpenCVConfig.cmake` file

* create folder `tracking_module/cmodules/build` and run `cmake ..` from there

* Open `MDP.sln` in Visual Studio, change build type to `Release` from `Debug`, right click on `All Build` in the Solution Explorer and select `Build`

* Once this completes, right click on `INSTALL` in the Solution Explorer and select `Build`









