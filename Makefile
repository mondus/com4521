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
################################################################################

# Project folders
PROJECTS=Lab03_Exercise01 Lab03_Exercise02

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
