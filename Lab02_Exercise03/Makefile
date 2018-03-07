################################################################################
#
# Simple Makefile script for COM4521/COM6521
#
# Authors : Dr Paul Richmond
#
# NOTICE TO USER:
#
# This is an example Makefile for building a single source exercise from the lab 
# classes exercise in COM4521/COM6521 labs.
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
#
################################################################################

# Change the example variable to build a different source module (e.g. hello/example1/example4)
EXAMPLE=exercise03_sln

# Makefile variables 
# Add extra targets to OBJ with space separator e.g. If there is as source file random.c then add random.o to OBJ)
# Add any additional dependancies (header files) to DEPS. e.g. if there is aheader file random.h required by your source modules then add this to DEPS.
CC=gcc
CFLAGS= -O3 -Wextra
OBJ=$(EXAMPLE).o
DEPS=random.h

# Build rule for object files ($@ is left hand side of rule, $< is first item from the right hand side of rule)
%.o : %.c $(DEPS)
	$(CC) -c -o $@ $< $(CFLAGS)

# Make example ($^ is all items from right hand side of the rule)
$(EXAMPLE) : $(OBJ)
	$(CC) -o $@ $^ $(CFLAGS)

# PHONY prevents make from doing something with a filename called clean
.PHONY : clean
clean:
	rm -rf $(EXAMPLE) $(OBJ)
