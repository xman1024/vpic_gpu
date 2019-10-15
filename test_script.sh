#!/bin/bash

set -e

mkdir -p build/
cd build
../arch/gcc/reference-Release
make -j
./bin/vpic ../sample/lyin_sequoia_1
mpirun -np 1 nvprof ./lyin_sequoia_1.Linux
