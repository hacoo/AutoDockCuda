/* Henry Cooney <hacoo36@gmail.com> <Github: hacoo>
   AutoDockCuda: <https://github.com/hacoo/AutoDockCuda>
   
   Defines structs for working with GPU memory.

*/

#include "typedefs.h"
#ifndef CUDA_STRUCTS_H
#define CUDA_STRUCTS_H
#endif
#ifndef _STRUCTS_H
#include "structs.h"
#endif


// This struct contains all pointers allocated to the gpu
// EXCEPT those allocated to constant memory


// Pointers related to the internal energy lookup tablen
typedef struct cuda_energytable {
  Real* e_vdW_Hb;  // vdW & Hb energiesx
  Real* sol_fn;
  Real* epsilon_fn;
  Real* r_epsilon_fn;
} Cuda_Energytable;


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
  int* state_size_dev; // the number of parameters in each states
  
  // Pointers for eintcal
  double* internal_energies_dev; // results of internal energy calculation (eintcal)
  Cuda_Energytable etab; // energy table for eintcal
  NonbondParam* nonbondlist; 
  EnergyComponent* group_energy; // Could be null if not used
  int* Nnb_array; // Holds the number of nonbonds in each section (ligand, receptor, etc)
  int Nnb; // Number of nonbonds 
  int B_calcIntElec; 
  int B_include_1_4_interactions;
  int B_use_non_bond_cutoff;
  int B_have_flexible_residues;
  Real* qsp_abs_charges;
  Real scale_1_4;  
} CudaPtrs;
