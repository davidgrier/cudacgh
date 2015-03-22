# cudacgh

**CUDA-accelerated HOT hologram calculation for IDL**

IDL is the Interactive Data Language, and is a product of
[Exelis Visual Information Solutions](http://www.exelisvis.com)

CUDA is the Compute Unified Device Architecture and is a
product of
[NVIDIA](http://www.nvidia.com)

*cudacgh* is licensed under the GPLv3.

## What it does

This package provides an IDL interface for
calculating holograms for holographic optical trapping systems.

To use this package, make sure that your `IDL_PATH` includes
`/usr/local/IDL/cuda`.

This package is written and maintained by David G. Grier
(david.grier@nyu.edu)

## INSTALLATION

### Install the CUDA runtime and development packages

Information is available at https://developer.nvidia.com

### Install cudacgh

1. unpack the distribution in a convenient directory.
2. `cd cudacgh`
3. `make`
4. `make install`

Installation requires super-user priviledges.

## UNINSTALLATION

1. `cd cudacgh`
2. `make uninstall`
