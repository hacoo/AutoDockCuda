/* Henry Cooney <hacoo36@gmail.com> <Github: hacoo>
   AutoDockCuda: <https://github.com/hacoo/AutoDockCuda>
   
   Defines structs for working with GPU memory.

*/

#ifndef CUDA_STRUCTS_H
#define CUDA_STRUCTS_H
#endif


// This struct contains all pointers allocated to the gpu
// EXCEPT those allocated to constant memory
typedef struct cudaptrs {
  int* torsion_root_list_dev;
  char* atom_strings_dev;
  double* atom_crds_dev;
  int* natoms_dev;
  double* torsions_dev;
  double* states_dev; 
} CudaPtrs;
