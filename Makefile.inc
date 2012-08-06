# The definitions in this file should be changed to match your system

# CUDA_DIR   = /usr/local/cuda
# CUDA_DIR   = /opt/local/cuda
CUDA_DIR   = /opt/local/cuda4.2
#THRUST_DIR = $(HOME)/code/c_cpp/common/thrust-1.5.1
THRUST_DIR = $(HOME)/code/c_cpp/common/thrust-1.6

# LIB_ARCH   = lib
LIB_ARCH   = lib64
# GPU_ARCH   = sm_13
GPU_ARCH   = sm_20

GCC        = gcc
GXX        = g++
AR         = ar
NVCC       = $(CUDA_DIR)/bin/nvcc
DOXYGEN    = doxygen
RM         = rm
ECHO       = echo
MKFLAGS    = 

DEDISP_DEBUG = 0

#.SILENT :
