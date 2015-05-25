######################################################
# Build script for project
######################################################

# Set the example name
CI_EXAMPLE       := cudaAccuracyTest
# Executable name
EXECUTABLE       := $(CI_EXAMPLE)
# Cuda source files (compiled with cudacc)
CUFILES          := $(CI_EXAMPLE)_kernel.cu
CUDEPS           := 
# C/C++ source files (compiled with gcc / c++)
CCFILES          := $(CI_EXAMPLE).cpp

USEGLLIB         := 1
USEGLUT          := 1
USERENDERCHECKGL := 1

######################################################
# Rules and targets

INCLUDES         += -I../../code
ROOTBINDIR       ?= ../../bin
ROOTDIR          ?= /Developer/GPU\ Computing/C/src

include $(ROOTDIR)/../common/common.mk
