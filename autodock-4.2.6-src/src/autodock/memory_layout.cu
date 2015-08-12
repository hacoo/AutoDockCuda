#include "constants.h"
#include "typedefs.h"
#include <string.h>
#ifndef CUDA_HEADERS
#include "/pkgs/nvidia-cuda/5.5/include/cuda.h"
#include "/pkgs/nvidia-cuda/5.5/include/cuda_runtime.h"
#endif
#include <stdio.h>
#include <stdlib.h>
#ifndef _SUPPORT_H
#include "support.h"
#endif
#ifndef CUDA_UTILS_HOST_H
#include "cuda_utils_host.h"
#endif
#ifndef _STRUCTS_H
#include "structs.h"
#endif
#ifndef _AUTOCOMM
#include "autocomm.h"
#endif
#ifndef GPU_VARIABLES_H
#include "gpu_variables.h"
#endif
#ifndef CUDA_STRUCTS_H
#include "cuda_structs.h"
#endif



//const int ATOM_SIZE = (6 + MAX_TORS) * 3 * sizeof(Real);
//const int MOL_INDV_SIZE = (7 + MAX_TORS) * sizeof(Real) + MAX_ATOMS * ATOM_SIZE;


/////////////////////***********************************************/////////////////////
///****   THESE ARE THE UTILITY FUNCTIONS TO USE TO ACCESS DATA ON THE GPU    *****/////
/*
__device__ Real * getIndvAttribute(int idx) {
	//all data is packed into array in x,y,z,qw,qx,qy,qz, [torsion data], ......
	//returns the start address, move to next item by adding sizeof(Real)
	return globalReals + (idx * MOL_INDV_SIZE) * sizeof(Real);
	}*/
/*
__device__ Real * getTorsion(int indvIdx, int torsionIdx) {
	//all data is packed into array in x1,y1,z1,theta1, x2,y2, .....
	//returns the start address, move to next item by adding sizeof(Real)
	return globalReals + (indvIdx * MOL_INDV_SIZE + 7 + 4 * torsionIdx) * sizeof(Real);
	}*/
/*
__device__ char*  getAtom(int indvIdx, int atom) {
	//all data is packed into array in c11,c12,...c1MAX_CHARS, c21, c22, c23, ...
	//returns the start address, move to next item by adding sizeof(char)
  return (char*) (globalChars + (indvIdx * MAX_TORS * MAX_CHARS + atom * MAX_CHARS) * sizeof(char));
}
*/


bool allocate_pop_to_gpu(Population& pop_in, int ntors, CudaPtrs* ptrs) {
  
  int pop_size = pop_in.num_individuals();
  int state_size = 10 + ntors; // The total number of items in each STATE item
  // - 3 trans coords + 4 quat coords + 3 center coords + ntors torsions
  int i, ii;
  Molecule* first_mol = pop_in[0].mol; 
  State current_state; 

  int natoms = getNumAtoms(first_mol);  
  double* atom_crds = getAtomCrds(first_mol);
  double* torsions = getTorsions(first_mol, ntors);
  int* torsion_root_list = getTorsionRootList(first_mol, ntors); // List of torsion root atoms
  char* atom_strings = getAtomStringArray(first_mol); // ragged array of atom strings
  
  double states[pop_size * state_size]; // flat array of individual states
  
  // Constant memory -- not implemented yet
  //  gpuErrchk(cudaMemcpyToSymbol(atom_crds_dev, atom_crds, sizeof(double)*natoms*SPACE));
  // gpuErrchk(cudaMemcpyToSymbol(torsions_dev, torsions,  sizeof(double)*ntors*SPACE));
  /// gpuErrchk(cudaMemcpyToSymbol(natoms_dev, &natoms, sizeof(int)));


  gpuErrchk(cudaMalloc((void**) &(ptrs->natoms_dev),
		       sizeof(int)));
  gpuErrchk(cudaMemcpy(ptrs->natoms_dev, &natoms, 
		       sizeof(int), cudaMemcpyHostToDevice));

  gpuErrchk(cudaMalloc((void**) &(ptrs->ntors_dev),
		       sizeof(int)));
  gpuErrchk(cudaMemcpy(ptrs->ntors_dev, &ntors, 
		       sizeof(int), cudaMemcpyHostToDevice));

  gpuErrchk(cudaMalloc((void**) &(ptrs->state_size_dev),
		       sizeof(int)));
  gpuErrchk(cudaMemcpy(ptrs->state_size_dev, &state_size, 
		       sizeof(int), cudaMemcpyHostToDevice));



  gpuErrchk(cudaMalloc((void**) &(ptrs->atom_crds_dev),
		       sizeof(double)*natoms*SPACE));
  gpuErrchk(cudaMemcpy(ptrs->atom_crds_dev, atom_crds, 
		       sizeof(double)*natoms*SPACE, cudaMemcpyHostToDevice));

  gpuErrchk(cudaMalloc((void**) &(ptrs->torsions_dev),
		       sizeof(double)*ntors*SPACE));
  gpuErrchk(cudaMemcpy(ptrs->torsions_dev, torsions, 
		       sizeof(double)*ntors*SPACE, cudaMemcpyHostToDevice));
  
  gpuErrchk(cudaMalloc((void**) &(ptrs->torsion_root_list_dev),
		       sizeof(int)*ntors*natoms));
  gpuErrchk(cudaMemcpy(ptrs->torsion_root_list_dev, torsion_root_list, 
		       sizeof(int)*natoms*ntors, cudaMemcpyHostToDevice));
  
  gpuErrchk(cudaMalloc((void**) &(ptrs->atom_strings_dev), 
		       sizeof(char)*natoms*MAX_CHARS));
  for(i=0; i<natoms; ++i){
    gpuErrchk(cudaMemcpy(ptrs->atom_strings_dev+i*MAX_CHARS, atom_strings+i*MAX_CHARS,
	      MAX_CHARS,
	      cudaMemcpyHostToDevice));
  }

  // TODO: Set constant memory addresses in ptr struct -- how do I make this work?
  //gpuErrchk(cudaGetSymbolAddress((void**)&(ptrs->atom_crds_dev), atom_crds_dev));
  

  for (int i = 0; i < pop_size; ++i) {
    current_state = pop_in[i].phenotyp.make_state(ntors);
    // Translation:
    states[i*state_size] = current_state.T.x;
    states[i*state_size+1] = current_state.T.y;
    states[i*state_size+2] = current_state.T.z;
    // Quaternion
    states[i*state_size+3] = current_state.Q.w;
    states[i*state_size+4] = current_state.Q.x;
    states[i*state_size+5] = current_state.Q.y;
    states[i*state_size+6] = current_state.Q.z;
    // Center
    states[i*state_size+7] = current_state.Center.x;
    states[i*state_size+8] = current_state.Center.y;
    states[i*state_size+9] = current_state.Center.z;
    // Torsions
    for (ii=0; ii<ntors; ++ii) {
      states[i*state_size+10+ii] = current_state.tor[ii];
    }
  }

    
  // Allocate array of individual states -- it is a flat array containing 
  // the translation, rotation, and torsions of each individual
  gpuErrchk(cudaMalloc((void**) &(ptrs->states_dev),
		       sizeof(double)*pop_size*state_size));
  gpuErrchk(cudaMemcpy(ptrs->states_dev, states, 
		       sizeof(double)*state_size*pop_size, cudaMemcpyHostToDevice));
  
  // Allocate array of inidividual atom coordinates -- also a flat array.
  // This array starts out initialized to 0 and is filled as calculations progress.
  gpuErrchk(cudaMalloc((void**) &(ptrs->indiv_crds_dev),
		       sizeof(double)*pop_size*natoms*SPACE));
  gpuErrchk(cudaMemset(ptrs->indiv_crds_dev, 0x00, 
		       sizeof(double)*pop_size*natoms*SPACE));
  
  
  free(atom_crds);
  free(atom_strings);
  free(torsions);
  free(torsion_root_list);
  
  return true;
}
