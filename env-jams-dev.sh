#!/bin/bash

# picked up by JAMS/Cmake
# 80 -> Ampere (A30)
# 89 -> Ada Lovelace (L40s)
# 90 -> Hopper (H200)
# 120 -> Blackwell (RTX 5080/5090/6000)
export CUDA_ARCHITECTURES=120

spack env activate jams-dev
spack load gcc@13.4.0
