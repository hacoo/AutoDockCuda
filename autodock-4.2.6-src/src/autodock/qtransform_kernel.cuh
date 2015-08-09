/* Henry Cooney <hacoo36@gmail.com> <Github: hacoo>
   AutoDockCuda: <https://github.com/hacoo/AutoDockCuda>
   
   qtransform_kernel.cuh
   
   Header file for qtransform_kernel.cu. qtransform is responsible for 
   applying rotation and translation to all atoms in a ligand.

   This kernel should be run on a population array already
   stored on the GPU. After it is run, the coordinates of each
   atom in each individual will now be updated to reflect
   their translation and quaternion rotation.

*/

#ifndef QTRANSFORM_KERNEL_H
#define QTRANSFORM_KERNEL_H
#endif
#ifndef CUDA_STRUCTS_H
#include "cuda_structs.h"
#endif


__global__
void qtransform_kernel(CudaPtrs ptrs) {
  // Quaternion transformation kernel. Does rigid-body
  // translation and rotation for each individual configuration.
  // Each block is responsible for one individual; each
  // thread handles one atom.
  
  // Each block should have natoms threads, or 12 threads,
  // whichever is greater (at least 12 threads are needed 
  // to load all data to SM).

  // Based on qtransform.cc (Autodock 4.2.6 equivalent)

  __shared__ double QT[7]; // quaternion and translation --
    // x, y, z qw, qx, qy, qz
  __shared__ double r[9]; // these are precomputed constants to each coordinate
    

    
  
  
}
