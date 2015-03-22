#
# Makefile for cudacgh
#
# Modification History
# 03/07/2015 Written by David G. Grier, New York University
#
# Copyright (c) 2015 David G. Grier
#
TARGET = cudacgh
INSTDIR = /usr/local/IDL/cuda

IDL = idl -quiet
INSTALL = install

SRC = $(TARGET).cu

OS = $(shell uname -s | tr '[:upper:]' '[:lower:]')
ARCH = $(OS).$(shell uname -m)

LIBRARY = $(TARGET).$(ARCH).so
DLM = $(TARGET).dlm

ifeq ($(OS),darwin)
	IDLDIR = /Applications/exelis/idl
endif

ifeq ($(OS),linux)
	IDLDIR = /usr/local/exelis/idl
endif

IDLINCS = -I$(IDLDIR)/external/include
IDLLIBS = -L$(IDLDIR)/bin/bin.$(ARCH) -lidl

INCS = $(IDLINCS)
LIBS = $(IDLLIBS)

CFLAGS = -O3 -Xcompiler -fPIC $(INCS)
LDFLAGS = --shared $(LIBS)

NVCC = nvcc

all: $(LIBRARY)

$(LIBRARY): $(SRC) $(DLM)
	$(NVCC) $(CFLAGS) $(LDFLAGS) -o $(LIBRARY) $(SRC)

install: $(LIBRARY) $(DLM)
	sudo $(INSTALL) -d $(INSTALLDIR)
	sudo $(INSTALL) $(LIBRARY) $(DLM) $(INSTALLDIR)

uninstall:
	sudo -rm $(INSTALLDIR)/{$(LIBRARY),$(DLM)}

clean:
	-rm *.o *.so
