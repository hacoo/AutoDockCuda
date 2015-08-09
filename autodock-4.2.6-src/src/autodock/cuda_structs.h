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
  int* torsion_root_list_dev; // tors root list - represents torsion tree
  char* atom_strings_dev; // string representing atom characteristics
  double* atom_crds_dev; // starting coords of each atom
  int* natoms_dev; // number of atoms
  double* torsions_dev; // number of torsions
  double* states_dev; // each individual's 'state' (translation, quat, etc)
  double* indiv_crds_dev; // each individual's atom coordinates -- filled in by qtransform
  // and torsion kernels
  int* ntors_dev;// number of torsions
} CudaPtrs;
