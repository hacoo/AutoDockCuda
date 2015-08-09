
/* Henry Cooney <hacoo36@gmail.com> <Github: hacoo>
   AutoDockCuda: <https://github.com/hacoo/AutoDockCuda>

   start_CUDA.cu
   
   This file's job is to start the actual CUDA kernel. It
   should call needed kernels in order.

   This file also defines gpu memory pointers. Because CUDA
   is picky about how files are #included together, you should 
   #include all kernels after the gpu memory pointer decs.
   
*/

#ifndef _SUPPORT_H
#include "support.h"
#endif
#ifndef CUDA_HEADERS
#include "/pkgs/nvidia-cuda/5.5/include/cuda.h"
#include "/pkgs/nvidia-cuda/5.5/include/cuda_runtime.h"
#endif
#include "cuda_utils_host.h"

#include "gpu_variables.h"
#include "memory_layout.cuh"
#include "cuda_utils_gpu.cuh"



void start_CUDA_on_population(Population* this_pop, int ntors) {
  // Begins evaluation of Population on GPU. For now, this is a 
  // placeholder that will allow kernels to be tested, etc.
  

  int natoms = getNumAtoms((*this_pop)[0].mol);
  printf("Allocating %d atoms and %d torsions to GPU...\n", natoms, ntors);
    
  allocate_pop_to_gpu(*this_pop, ntors);
  
  dim3 dimBlock(natoms,1,1);
  dim3 dimGrid(2,1,1);

  printAutoDockMemoryKernel<<<dimGrid, dimBlock>>>();
  cudaError_t err = cudaGetLastError();
  if (err != cudaSuccess) 
    printf("Error: %s\n", cudaGetErrorString(err));

  printf("Done! \n");  
}

