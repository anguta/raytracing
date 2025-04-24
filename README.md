# CPU/GPU Performance Analysis of Ray Tracing
The following repository contains the work for perfomance analysis of the Ray Tracing algorithm on the CPU and GPU devices.


# How to Run
*Make sure you are in the gmclab station signed in with your ssh*

CPU: cd src; g++ -std=c++17 CPU/Main.cpp -o main_cpp; ./main_cpp > cpu.ppm

GPU Naive: cd src; nvcc -arch=sm_75 -O0 GPU/main_naive.cu -o main_naive; ./main_naive > naive.ppm

GPU Optimize: cd src; nvcc -arch=sm_75 -O0 GPU/Main2.cu -o main; ./main > optimized.ppm

# PPM to JPG
I used https://www.freeconvert.com/ppm-to-jpg to convert the ppm to jpg
