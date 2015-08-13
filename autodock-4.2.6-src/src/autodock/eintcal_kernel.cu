/* Henry Cooney <hacoo36@gmail.com> <Github: hacoo>
   AutoDockCuda: <https://github.com/hacoo/AutoDockCuda>
   
   eintcal_kernel.cu

   CUDAfied version of eintcal. eintcal does internal energy calculation
   for a ligand.

   Internal energy is the energy of the ligand due to the configuration of 
   the ligand (i.e. how it is bent or flexed). 
   
   Each atom in the ligand contributes to the overall internal energy.
   Additionally, if the receptor is flexible, the receptor's internal
   energy will be calculated as well (not implemented yet).
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
#ifndef _STRUCTS_H
#include "struct.h"
#endif

__global__
void eintcal_kernel(CudaPtrs ptrs) {
  // Does internal energy calculation for a population array
  // which is already on the GPU.

  // Code adapted from eintcal.cc.

  // Parallelizes eintcal by assigning a thread to each nonbond.
  // 
  // Blocks are made of up to 512 threads. Each row of the grid
  // represents on individual. Each row may be up to 1024 blocks
  // (depending on the number of nonbonds.)

  // This kernel does NOT differentiate between ligand/receptor energies,
  // it calculates the total ONLY.

  // energy of each atom; will be reduced at end of kernel
  __shared__ double energies[512];
  
  
  const double nbc2 = ptrs.B_use_non_bond_cutoff ? NBC2 : 999 * 999;
  register int idx = threadIdx.x;
  register int nonbond_index = blockIdx.x*blockDim.x + threadIdx.x;
  register int Nnb = ptrs.Nnb;
  double e_total = 0.0; // total energy for THIS ATOM

  if (nonbond_index < Nnb) {
    
    register NonbondParam* nonbondlist = ptrs.nonbondlist;
    register double* crd = ptrs.indiv_crds_dev;
    register int natoms = *(ptrs.natoms_dev);
  
    int nonbond_type, index, index_lt_NDIEL, index_lt_NEINT;
    double dx, dy, dz, r2;
    double nb_desolv, e_desolv;
    int a1, a2;

    // First, get information from the nonbondlist
    a1 = nonbondlist[nonbond_index].a1;
    a2 = nonbondlist[nonbond_index].a2;
    
    dx = crd[blockIdx.y*natoms*SPACE + a1*SPACE];
    dy = crd[blockIdx.y*natoms*SPACE + a1*SPACE+1];
    dx = crd[blockIdx.y*natoms*SPACE + a1*SPACE+2];

    r2 = dx*dx + dy*dy + dz*dz;
    r2 = clamp(r2, (RMIN_ELEC*RMIN_ELEC)); //macro defined in constants.h
    
#ifndef NOSQRT     // Use square-root, slower...
    index = Ang_to_index(sqrt(r2)); 
#else             //  Non-square-rooting version, faster...
    index = SqAng_to_index(r2);
#endif  // NOSQRT

    index_lt_NDIEL = BoundedNdiel(index);
    nonbond_type = nonbondlist[nonbond_index].nonbond_type;
    nb_desolv = nonbondlist[nonbond_index].desolv;

    if (ptrs.B_calcIntElec) {
      //  Calculate  Electrostatic  Energy
      double r_dielectric = ptrs.etab.r_epsilon_fn[index_lt_NDIEL];
      e_total += nonbondlist[nonbond_index].q1q2 * r_dielectric;
    }
    
    e_desolv = ptrs.etab.sol_fn[index_lt_NDIEL] * nb_desolv;
    if  ( r2 < nbc2 ) {   
      int t1, t2; 
      t1 = nonbondlist[nonbond_index].t1; // t1 is a map_index
      t2 = nonbondlist[nonbond_index].t2; // t2 is a map_index
      index_lt_NEINT = BoundedNeint(index);  // guarantees that index_lt_NEINT is never greater than (NEINT - 1) (scaled NBC, non-bond cutoff)
      double e_vdW_Hb= ptrs.etab.e_vdW_Hb[index_lt_NEINT*MAX_ATOM_TYPES*MAX_ATOM_TYPES
		 +t2*MAX_ATOM_TYPES + t1];
      if (ptrs.B_include_1_4_interactions && nonbond_type==4 ) {
	//| Compute a scaled 1-4 interaction,
	e_vdW_Hb *= ptrs.scale_1_4;
	e_desolv *= ptrs.scale_1_4;
      }
      e_total += e_vdW_Hb + e_desolv;
    }
    else 
      e_total += e_desolv; // no NBC-based cutoff for desolvation  
    
  }
  
  energies[idx] = e_total; // will be 0.0 for overhanging threads
  
  // 
  
}


