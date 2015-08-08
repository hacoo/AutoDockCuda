
/* Henry Cooney <hacoo36@gmail.com> <Github: hacoo>
   AutoDockCuda: <https://github.com/hacoo/AutoDockCuda>

   cudat_utils_gpu.cuh
   
   Includes utility functions for dealing with CUDA and autodock, on the GPU side.   
   includes functions for examing data on the GPU.

*/



#ifndef CUDA_UTILS_GPU_H
#define CUDA_UTILS_GPU_H
#endif

__global__
void printAutoDockMemoryKernel(int* natoms_dev) ;
