################################################################################
#
# Simple Makefile script for COM4521/COM6521
#
# Authors : Dr Paul Richmond
#
# NOTICE TO USER:
#
# This is an example Makefile for building a single CUDA source exercise from the 
# lab classes exercise in COM4521/COM6521 labs.
#
# This makefile is NOT for building the code on Windows. If you are using Visual 
# Studio then you will need to create a Visual Studio project as directed by the
# lectures/lab exercise. The makefile is for providing an additional build option
# for user familar with Linux.
#
# You may need to modify this makefile for building your own code or for building 
# and linking multiple source modules.
#
# Makefile only supported on Linux Platforms.
# 
# IMPORTANT: To use this Makefile on ShARC you MUST ensure you have loaded the 
# appropriate modules. e.g.
#
# module load dev/gcc/4.9.4
# module load libs/CUDA
#
# You do NOT need to be on a GPU node in order to build GPU code but you must be 
# on a GPU node to run the code you have built. See the module website for more
# information.
#
################################################################################

# Change the example variable to build a different source module (e.g. hello/example1/example4)
EXAMPLE=nsight-gtc

# Makefile variables 
# Add extra targets to OBJ with space separator e.g. If there is as source file random.c then add random.o to OBJ)
# Add any additional dependencies (header files) to DEPS. e.g. if there is a header file random.h required by your source modules then add this to DEPS.
CC=gcc
CFLAGS= -O3 -Wextra -fopenmp
NVCC=nvcc
NVCC_FLAGS= -gencode arch=compute_35,code=compute_35
OBJ=$(EXAMPLE).o
DEPS=
step?=0x00

# Build rule for object files ($@ is left hand side of rule, $< is first item from the right hand side of rule)
%.o : %.cu $(DEPS)
	$(NVCC) -c -o $@ $< $(NVCC_FLAGS) $(addprefix -Xcompiler ,$(CCFLAGS)) -D OPTIMIZATION_STEP=$(step)

# Make example ($^ is all items from right hand side of the rule)
$(EXAMPLE) : $(OBJ)
	$(NVCC) -o $@ $^ $(NVCC_FLAGS) $(addprefix -Xcompiler ,$(CCFLAGS)) -D OPTIMIZATION_STEP=$(step)

# PHONY prevents make from doing something with a filename called clean
.PHONY : clean
clean:
	rm -rf $(EXAMPLE) $(OBJ)