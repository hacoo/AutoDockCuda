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

  /* __shared__ double QT[7]; // quaternion and translation --
    // x, y, z qw, qx, qy, qz
  __shared__ double r[9]; // these are precomputed constants to each coordinate
  */

  // Currently uses naive memory sharing -- each thread loads its own quaternion
  // parameters. Will test this version, and then improve incrementally.
  

  // NOT OPTIMIZED! Right now EVERY thread must load ALL relevant information.
  // Unfortunatley, this memory is laid out in a complicated way, so it will
  // take me some time to figure out how to share it efficiently.

  // UNOPTIMIZED
  //int tid = threadIdx.x;
  register int natoms = *ptrs.natoms_dev;
  register int state_size = *ptrs.state_size_dev;
  register int atom_index = threadIdx.x*SPACE;
  register int output_index = natoms*SPACE*blockIdx.x + threadIdx.x*SPACE;
  register int state_index = blockIdx.x*state_size;

  register double w, x, y, z;
  register double tx, ty, tz;
  register double omtxx;
  register double twx, txy, txz;
  register double twy, tyy, tyz;
  register double twz, tzz;
  register double r11, r12, r13, r21, r22, r23, r31, r32, r33;

  

  
  //   ptrs.indiv_crds_dev[natoms*SPACE*bid+ tid*SPACE] = 1.0;

  w = ptrs.states_dev[state_index+3];
  x = ptrs.states_dev[state_index+4];
  y = ptrs.states_dev[state_index+5];
  z = ptrs.states_dev[state_index+6];
  
  tx  = x+x;
  ty  = y+y;
  tz  = z+z;

  twx = w*tx;
  omtxx = 1. - x*tx;
  txy = y*tx;
  txz = z*tx;

  twy = w*ty;
  tyy = y*ty;
  tyz = z*ty;

  twz = w*tz;
  tzz = z*tz;

  r11 = 1. - tyy - tzz;
  r12 =      txy - twz;
  r13 =      txz + twy;
  r21 =      txy + twz;
  r22 = omtxx    - tzz;
  r23 =      tyz - twx;
  r31 =      txz - twy;
  r32 =      tyz + twx;
  r33 = omtxx    - tyy;
  // END UNOPTIMIZED SECION

  // Each thread writes 3 coordinates for the corresponding atom:
  
  ptrs.indiv_crds_dev[output_index] = 
    ptrs.atom_crds_dev[atom_index]*r11 + 
    ptrs.atom_crds_dev[atom_index+1]*r21 +
    ptrs.atom_crds_dev[atom_index+2]*r31 +
    ptrs.states_dev[state_index];
  
  ptrs.indiv_crds_dev[output_index+1] = 
    ptrs.atom_crds_dev[atom_index]*r12 +
    ptrs.atom_crds_dev[atom_index+1]*r22 +
    ptrs.atom_crds_dev[atom_index+2]*r32 +
    ptrs.states_dev[state_index+1];

  ptrs.indiv_crds_dev[output_index+2] = 
    ptrs.atom_crds_dev[atom_index]*r13 +
    ptrs.atom_crds_dev[atom_index+1]*r23 +
    ptrs.atom_crds_dev[atom_index+2]*r33 +
    ptrs.states_dev[state_index+2];
 
}

