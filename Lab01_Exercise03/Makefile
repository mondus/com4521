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

# Change the example variable to build a different source module
EXAMPLE=exercise03

# Makefile variables
CC=gcc
CFLAGS= -O3 -Wextra
OBJ=$(EXAMPLE).o random.o
DEPS=random.h

# Build rule for object files ($@ is left hand side of rule, $< is first item from the righthand side of rule)
%.o : %.c $(DEPS)
	$(CC) -c -o $@ $< $(CFLAGS)

# Make example ($^ is all items from right hand side of the rule)
$(EXAMPLE) : $(OBJ)
	$(CC) -o $@ $^ $(CFLAGS)

# PHONY prevents make from doing something with a filename called clean
.PHONY : clean
clean:
	rm -rf $(EXAMPLE) $(OBJ)
