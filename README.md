# yolov4-darkflow-ubuntu-customdataset
implementation of yolov4 on Ubuntu using custom data set

#Install the Darknet YOLO v4 training environment

1) download the zip file and extract it.

2) The first step in building/installing YOLO v4 on Ubuntu is installing its dependencies. Here I am assuming that you have a freshly installed Ubuntu 20.04 installation which is only having default installed packages.

   Note: If you have already installed the ROS framework, Open-CV, you don’t need to install it again as the prerequisites. It may break the existing packages.

# Installing YOLO Prerequisites

Here are the important prerequisites of YOLO.

We can run YOLO either in CPU or run with GPU acceleration. In order to do GPU acceleration, you may need a good Nvidia based graphics card with CUDA cores.

Use the following command to install the prerequisites, you can find the explanation of each one after that.

 CMake >= 3.8
 CUDA 10.0 (For GPU)
 OpenCV >= 2.4 (For CPU and GPU)
 cuDNN >= 7.0 for CUDA 10.0 (for GPU)
 OpenMP (for CPU)
 Other Dependencies: make, git, g++

# CMake >= 3.8 (for modern CUDA support)(CPU and GPU)

The Ubuntu 20.04 is having CMake version 3.16, we can install this using apt package manager.

Installing CMake in Ubuntu 20.04
 
 sudo apt install cmake 

For checking your CMake version, you can use the following command

 cmake --version
 
# CUDA 10.0 (For GPU) 

You can install CUDA if you have a GPU from NVIDIA, adding GPU for YOLO will speed up the object detection process.

CUDA is a parallel computing platform and application programming interface model created by Nvidia. It allows software developers and software engineers to use a CUDA-enabled graphics processing unit for general purpose processing.

Here are the tutorials to install CUDA 10 on Ubuntu.

Download and install CUDA 10 Toolkit

How to install CUDA on Ubuntu 20.04 Focal Fossa Linux
 
# OpenCV >= 2.4 (For CPU and GPU)

OpenCV is a library of programming functions mainly aimed at real-time computer vision. Originally developed by Intel, it was later supported by Willow Garage then Itseez. The library is cross-platform and free for use under the open-source BSD license.

Note: The OpenCV is an optional install YOLO, but if you install it, you will get a window system to display the output image from YOLO detection. Otherwise, the output image from YOLO will be saved as an image file. I have enabled OpenCV for this tutorial so that you can see the output of YOLO in a window.

You can install OpenCV in Ubuntu using the apt package manager or using compiling the source code.

# Installing OpenCV using package manager

Here is the command to quickly install OpenCV and its Python extension using the apt package manager.

    sudo apt install libopencv-dev python3-opencv   

After installing OpenCV, you can find the existing version of OpenCV using the following command

    opencv_version

# cuDNN >= 7.0 for CUDA 10.0 (for GPU)

The NVIDIA CUDA® Deep Neural Network library (cuDNN) is a GPU-accelerated library of primitives for deep neural networks. cuDNN provides highly tuned implementations for standard routines such as forward and backward convolution, pooling, normalization, and activation layers.

Here is the installation procedure for cuDNN.

 

  https://docs.nvidia.com/deeplearning/sdk/cudnn-archived/cudnn_741/cudnn-install/index.html
  
  
# Other Dependencies (For CPU and GPU)  
  
  Here are other dependencies used for CPU and GPU configuration.
  
     sudo apt install make git g++    
     
     
# Building YOLO v4 using Make     
 
  Switch to the darknet folder after download. Open the Makefile in the darknet folder. You can see some of the variables at the beginning of the Makefile. If you want to compile darknet for CPU, you can use the following flags.
 
 # For CPU build
 
Set AVX=1 and OPENMP=1 to speedup on CPU (if an error occurs then set AVX=0), 
Set LIBSO=1 will create the shared library of the darknet, ‘libdarknet.so‘, which is used to interface darknet and Python.
Set ZED_CAMERA=1 if you are working with ZED camera and its SDK

 GPU=0
 
 CUDNN=0
 
 CUDNN_HALF=0
 
 OPENCV=1
 
 AVX=1
 
 OPENMP=1
 
 LIBSO=1
 
 ZED_CAMERA=0
 
 ZED_CAMERA_v2_8=0 
 
 # For GPU build
 
 set GPU=1 and CUDNN=1 to speedup on GPU

 set CUDNN_HALF=1 to further speedup 3 x times (Mixed-precision on Tensor Cores) GPU: Volta, Xavier, Turing and higher
 
 GPU=1
 
 CUDNN=1
 
 CUDNN_HALF=1
 
 OPENCV=1
 
 AVX=0
 
 OPENMP=0
 
 LIBSO=1
 
 ZED_CAMERA=0
 
 ZED_CAMERA_v2_8=0 
 
 After doing these changes, just execute the following command from the darknet folder.
 
   make
 
 After build, you can able find darknet and libdarknet.so in the build path.
 
# How to train (to detect your custom objects):

1) For training cfg/yolov4-custom.cfg download the pre-trained weights-file (162 MB): https://github.com/AlexeyAB/darknet/releases/download/darknet_yolo_v3_optimal/yolov4.conv.137 

2) After downloading the weight file locate it in the root folder of darknet that is extracted few steps before.

3) Create file yolo-obj.cfg with the same content as in yolov4-custom.cfg (or copy yolov4-custom.cfg to yolo-obj.cfg) and:

   a) change line batch to batch=64

   b) change line subdivisions to subdivisions=16

   c) change line max_batches to (classes*2000 but not less than number of training images, but not less than number of training images and not less than 6000), f.e. max_batches=6000 if you train for 3 classes

   d) change line steps to 80% and 90% of max_batches, f.e. steps=4800,5400

   e) set network size width=416 height=416 or any value multiple of 32: 
      https://github.com/AlexeyAB/darknet/blob/0039fd26786ab5f71d5af725fc18b3f521e7acfd/cfg/yolov3.cfg#L8-L9

   f) change line classes=80 to your number of objects in each of 3 [yolo]-layers:

      https://github.com/AlexeyAB/darknet/blob/0039fd26786ab5f71d5af725fc18b3f521e7acfd/cfg/yolov3.cfg#L610
      https://github.com/AlexeyAB/darknet/blob/0039fd26786ab5f71d5af725fc18b3f521e7acfd/cfg/yolov3.cfg#L696
      https://github.com/AlexeyAB/darknet/blob/0039fd26786ab5f71d5af725fc18b3f521e7acfd/cfg/yolov3.cfg#L783
      
   g) change [filters=255] to filters=(classes + 5)x3 in the 3 [convolutional] before each [yolo] layer, keep in mind that it only has to be the last [convolutional] before each of the [yolo] layers.
   
      https://github.com/AlexeyAB/darknet/blob/0039fd26786ab5f71d5af725fc18b3f521e7acfd/cfg/yolov3.cfg#L603
      https://github.com/AlexeyAB/darknet/blob/0039fd26786ab5f71d5af725fc18b3f521e7acfd/cfg/yolov3.cfg#L689
      https://github.com/AlexeyAB/darknet/blob/0039fd26786ab5f71d5af725fc18b3f521e7acfd/cfg/yolov3.cfg#L776
      
   h) when using [Gaussian_yolo] layers, change [filters=57] filters=(classes + 9)x3 in the 3 [convolutional] before each [Gaussian_yolo] layer
   
      https://github.com/AlexeyAB/darknet/blob/6e5bdf1282ad6b06ed0e962c3f5be67cf63d96dc/cfg/Gaussian_yolov3_BDD.cfg#L604
      https://github.com/AlexeyAB/darknet/blob/6e5bdf1282ad6b06ed0e962c3f5be67cf63d96dc/cfg/Gaussian_yolov3_BDD.cfg#L696
      https://github.com/AlexeyAB/darknet/blob/6e5bdf1282ad6b06ed0e962c3f5be67cf63d96dc/cfg/Gaussian_yolov3_BDD.cfg#L789
      
   i) So if classes=1 then should be filters=18. If classes=2 then write filters=21.
   
      (Do not write in the cfg-file: filters=(classes + 5)x3)

      (Generally filters depends on the classes, coords and number of masks, i.e. filters=(classes + coords + 1)*<number of mask>, where mask is indices of anchors. If mask is absence, then filters=(classes + coords + 1)*num)

    j) So for example, for 2 objects, your file yolo-obj.cfg should differ from yolov4-custom.cfg in such lines in each of 3 [yolo]-layers:
    
       [convolutional]
       filters=21

       [region]
       classes=2
       
    k) Create file obj.names in the root directory ,and write the object name or label name or class name which you want to train.
    
    l) Create file obj.data in the root directory , containing (where classes = number of objects):
    
    
       classes= 2
       train  = train.txt
       valid  = test.txt
       names =  obj.names
       backup = backup/
       
     m) Put image-files (.jpg) of your objects in the directory build\darknet\x64\data\obj\ (i think you have to create the new folder obj if it is not available.)
     
     n) You should label each object on images from your dataset. Use this visual GUI-software for marking bounded boxes of objects and generating annotation files.
     
        follow this tool https://github.com/tzutalin/labelImg(you should label the data and save it in yolo formate dont forget)
        
        For example for img1.jpg you will be created img1.txt
        
     o) txt file and the jpg file have to save in same folder together. 
     
     p) you have to create the train.txt and test .txt file ,which contains the location of every training and texting image
     
     just run the following command in the folder through the terminal you will get it and then locate these 2 txtx file in the root folder.
     
     for training.txt
     
         ls "$PWD/"*.jpg | head -100 > train.txt
         
         
     for testing.txt
     
         ls "$PWD/"*.jpg | tail -20 > test.txt
         
     q) To train on Linux use command: ./darknet detector train data/obj.data yolo-obj.cfg yolov4.conv.137 (just use ./darknet instead of darknet.exe)

 
# testing code

I will upload it soon 
