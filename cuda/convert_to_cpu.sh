#!/bin/bash
# Quick conversion script for CPU/GPU compatibility

for file in fields.cu boundaries.cu; do
    echo "Converting $file..."
   
    # Backup
    cp $file ${file}.bak
    
    # Replace CUDA includes with platform.h
    sed -i 's|#include <cuda_runtime.h>|#include "../include/platform.h"|' $file
    sed -i 's|#include <device_launch_parameters.h>||' $file
    
    # Replace __device__ and __global__
    sed -i 's/__device__/DEVICE_HOST/g' $file
    sed -i 's/__global__/GLOBAL/g' $file
    sed -i 's/__restrict__//g' $file
    
    echo "Converted $file (backup at ${file}.bak)"
done

echo "Done! Manual review required for kernel launch sites."
