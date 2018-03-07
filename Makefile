################################################################################
#
# Recursive Makefile script for COM4521/COM6521
#
# Authors : Dr Paul Richmond
#
# NOTICE TO USER:
#
# This is an example recursive Makefile for building a all exercises for a lab 
# class in COM4521/COM6521.
#
#
# Makefile only supported on Linux Platforms.
# 
# IMPORTANT: To use this Makefile on ShARC you MUST ensure you have loaded the 
# appropriate modules. e.g.
#
# module load dev/gcc/4.9.4
#
################################################################################

# Project folders
PROJECTS=Lab02_Exercise01 Lab02_Exercise02 Lab02_Exercise03 Lab02_Exercise04

.PHONY : all $(PROJECTS)
all: $(PROJECTS)

$(PROJECTS):
	$(MAKE) -C $@

# Clean
SUBCLEAN = $(addsuffix .clean,$(PROJECTS))

# PHONEY prevents make from doing something with a filename called clean
.PHONY : clean
clean: $(SUBCLEAN)

$(SUBCLEAN): %.clean :
	$(MAKE) -C $* clean
