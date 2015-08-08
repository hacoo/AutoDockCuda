
/* Henry Cooney <hacoo36@gmail.com> <Github: hacoo>
   AutoDockCuda: <https://github.com/hacoo/AutoDockCuda>

   cudat_utils_gpu.cu
   
   Includes utility functions for dealing with CUDA and autodock, on the GPU side.   
   includes functions for examing data on the GPU.

*/
#include "constants.h"
#include "typedefs.h"
#ifndef CUDA_HEADERS
#include "/pkgs/nvidia-cuda/5.5/include/cuda.h"
#include "/pkgs/nvidia-cuda/5.5/include/cuda_runtime.h"
#endif
#include <stdio.h>
#include <stdlib.h>
#ifndef CUDA_HEADERS
#include "/pkgs/nvidia-cuda/5.5/include/cuda.h"
#include "/pkgs/nvidia-cuda/5.5/include/cuda_runtime.h"
#endif
#ifndef CUDA_GPU_VARIABLES_H
#include "cuda_gpu_variables.cuh"
#endif


__global__
void printAutoDockMemoryKernel(int* natoms_dev) {
  // Testing / diagnostic kernel, prints 
  // contents of GPU memory to confirm they have been successfully
  // allocated.

  int idx = threadIdx.x;
  printf("Hello from thread %d! \n ", idx);
  if(idx == 0) {
    int natoms_local = *natoms_dev;
    // Since its a print function, threadIdx 0 will do all the work, very slowly.

    printf("Num atoms: %d \n ", natoms_local);
  }

}
