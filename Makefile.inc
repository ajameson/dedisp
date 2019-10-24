# The definitions in this file should be changed to match your system

CUDA_PATH   ?= /usr/local/cuda
INSTALL_DIR ?= $(HOME)/linux_64

CUDA_DIR   ?= $(CUDA_PATH)
THRUST_DIR ?= $(CUDA_DIR)/include

GPU_ARCH   ?= sm_30

LIB_ARCH   = lib64

GCC        = gcc
GXX        = g++
AR         = ar
NVCC       = $(CUDA_DIR)/bin/nvcc #-Xptxas -abi=no
DOXYGEN    = doxygen
RM         = rm
ECHO       = echo
MKFLAGS    = 

DEDISP_DEBUG = 0

#.SILENT :
