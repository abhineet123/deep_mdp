# Python dependencies

```
pip3 install xlwt
```

# Installation in Linux

## imbalanced-dataset-sampler       @ Installation_in_Linux

```
cd models/thirdparty/imbalanced-dataset-sampler && python3 setup.py install

```

## libsvm       @ Installation_in_Linux

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

## C_modules       @ Installation_in_Linux

```
cd cmodules && mkdir build && cd build
 
cmake ..
 
sudo make install
```


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









