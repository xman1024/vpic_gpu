#!/bin/bash

mkdir -p build/
cd build
../arch/gcc/reference-Release
make -j
./bin/vpic ../sample/lyin_sequoia_1
mpirun -np 1 ./lyin_sequoia_1.Linux