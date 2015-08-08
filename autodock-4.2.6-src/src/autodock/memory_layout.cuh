/* Henry Cooney <hacoo36@gmail.com> <Github: hacoo>
   AutoDockCuda: <https://github.com/hacoo/AutoDockCuda>

   memory_layout.h
    
   Header file for memory_layout.cu, written by Patrick Romero <Github: patrick38894>
   
   memory_layout.cu includes functions for initializing a population on
   the GPU.

*/


#ifndef MEMORY_LAYOUT_H
#define MEMORY_LAYOUT_H
#endif

#ifndef CUDA_HEADERS
#include "/pkgs/nvidia-cuda/5.5/include/cuda.h"
#include "/pkgs/nvidia-cuda/5.5/include/cuda_runtime.h"
#endif
#ifndef _STRUCTS_H
#include "structs.h"
#endif



// Function prototypes
bool allocate_pop_to_gpu(Population& pop_in, int ntors);
__device__ Real * getIndvAttribute(int idx);
__device__ Real * getTorsion(int indvIdx, int torsionIdx);
__device__ char*  getAtom(int indvIdx, int atom);


