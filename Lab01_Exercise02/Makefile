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
################################################################################

# Change the example variable to build a different source module
EXAMPLE=exercise02

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

# PHONEY prevents make from doing something with a filename called clean
.PHONEY : clean
clean:
	rm -rf $(EXAMPLE) $(OBJ)
