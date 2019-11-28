# Vector Particle-In-Cell (VPIC) GPU port

This is a port of [vpic](https://github.com/lanl/vpic) to Nvidia GPU using CUDA.
For information about VPIC in general please refer to `README_ORIGINAL.md`,
since this file contains only information about the port.

This code was written for SC19 Student Cluster Competition and many decisions
made were based on the information we had about the input deck for the
competition.

# New requirements

The most important new dependency we added is nvcc. We also have a strict
requirement on cmake version - cmake-3.9.1 is required, newer versions cause
nvcc to break due to passing unrecognised threading options (at least with
pthreads). We also use [thrust](https://github.com/thrust/thrust).

Both nvcc and thrust are included in CUDA Toolkit.

# Build instructions

Disable tests (they don't compile currently) by setting
`-DENABLE_INTEGRATED_TESTS` to `OFF` in the script invoking cmake.

# Changes made

Currently only particle pipeline was ported to GPU. That means `species::p`
has been  renamed to `species::device_p0`, and the new name refers to an array
in the device memory. Vectorised pipelines in `species_advance` have been
removed.

All cuda-related code resides in `src/cuda/` directory. Using `src/cuda/utils.h`
requires `.cc` rather than `.c`, so some files have been renamed.

Collisions were removed since we were informed that they would not be used in
the input deck for the competition. Same goes for particle injection. Using
those in input deck may result in compilation errors, segmentation faults or
undefined behaviours.

# TODO

1. Fix integrated tests.
2. Allow particle injections.
3. Port collisions.
4. Fix all the input decks (this likely requires 2 and 3 to be completed)

