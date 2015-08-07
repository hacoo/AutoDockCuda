/* Henry Cooney <hacoo36@gmail.com> <Github: hacoo>
   AutoDockCuda: <https://github.com/hacoo/AutoDockCuda>
   
   qtransform_kernel.cu
   
   CUDA implementation of qtransform. qtransform is responsible for 
   applying rotation and translation to all atoms in a ligand.

   This kernel should be run on a population array already
   stored on the GPU. After it is run, the coordinates of each
   atom in each individual will now be updated to reflect
   their translation and quaternion rotation.

*/

#include "constants.h"
#include "typedefs.h"
#ifndef CUDA_HEADERS
#include "/pkgs/nvidia-cuda/5.5/include/cuda.h"
#include "/pkgs/nvidia-cuda/5.5/include/cuda_runtime.h"
#endif


__global__
void qtransform_kernel(Real* globalReals) {
  // Does translation and rotation of each atom coordinate in
  // globalReals. Each thread should handle one atom, and 
  // each block should handle one individual.

  int individual_num = blockIdx.x;
  int atom_num = threadIdx.x;

  
  

}


