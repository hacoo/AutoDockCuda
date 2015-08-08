#include "constants.h"
#include "typedefs.h"
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


const int ATOM_SIZE = (6 + MAX_TORS) * 3 * sizeof(Real);
const int MOL_INDV_SIZE = (7 + MAX_TORS) * sizeof(Real) + MAX_ATOMS * ATOM_SIZE;

__device__ Real * globalReals;
__device__ char * globalChars;


/////////////////////***********************************************/////////////////////
///****   THESE ARE THE UTILITY FUNCTIONS TO USE TO ACCESS DATA ON THE GPU    *****/////

__device__ Real * getIndvAttribute(int idx) {
	//all data is packed into array in x,y,z,qw,qx,qy,qz, [torsion data], ......
	//returns the start address, move to next item by adding sizeof(Real)
	return globalReals + (idx * MOL_INDV_SIZE) * sizeof(Real);
}

__device__ Real * getTorsion(int indvIdx, int torsionIdx) {
	//all data is packed into array in x1,y1,z1,theta1, x2,y2, .....
	//returns the start address, move to next item by adding sizeof(Real)
	return globalReals + (indvIdx * MOL_INDV_SIZE + 7 + 4 * torsionIdx) * sizeof(Real);
}

__device__ char*  getAtom(int indvIdx, int atom) {
	//all data is packed into array in c11,c12,...c1MAX_CHARS, c21, c22, c23, ...
	//returns the start address, move to next item by adding sizeof(char)
  return (char*) (globalChars + (indvIdx * MAX_TORS * MAX_CHARS + atom * MAX_CHARS) * sizeof(char));
}


bool allocate_pop_to_gpu(Population& pop_in, int ntors) {
  
  
  State curr;
  int pop_size = pop_in.num_individuals();
  int state_size = 10 + ntors; // The total number of items in each STATE item
  // - 3 trans coords + 4 quat coords + 3 center coords + ntors torsions

  
  // There should be only one ligand molecule - this will be allocated
  // first (hopfully to constant memory). Then, allocate each
  // individual's State.
  int i, ii;
  Molecule* first_mol = pop_in[0].mol; 
  Molecule* current_mol;
  int natoms = getNumAtoms(first_mol);
  printf("Number of atoms: %d \n", natoms);

  double* atom_crds = getAtomCrds(first_mol);
  char** atom_strings = getAtomStringArray(first_mol); // ragged array of atom strings
  double* torsions = getTorsions(first_mol, ntors);
  int* torsion_root_list = getTorsionRootList(first_mol, ntors); // List of torsion root atoms

  
  /*
  printf("Contents of atom_crds: \n");
  for (i=0; i<natoms; ++i) {
  printf(" %f %f %f \n", atom_crds[3*i], atom_crds[3*i+1], atom_crds[3*i+2]);
  }

  printf("Contents of atom string array: \n");
  for (i=0; i<natoms; ++i) {
  printf("  %s\n", atom_strings[i]);
  }

  printf("Torsions: \n");
  for (i=0; i<ntors; ++i) {
  printf("  %f %f %f \n", torsions[3*i], torsions[3*i+1], torsions[3*i+2]);
  }
  
  printf("Root List: \n");
  for (i=0; i<ntors; ++i) {
    for (ii=0; ii<natoms; ++ii) {
      printf("%d ", torsion_root_list[i*natoms + ii]);
    }
    printf("\n");
  }
  */

 
  print_molecule(first_mol);
	  
  printf("Allocating population of %d individuals to GPU... \n", pop_size);
  //gpuErrchk(cudaMalloc((void **) &out, pop_size * MOL_INDV_SIZE));
  //gpuErrchk(cudaMalloc((void **) &atoms, pop_size * MAX_ATOMS * MAX_CHARS));

  //TODO: First allocate molecule (just once). Then
  // allocate state for each individual.

  for (int i = 0; i < pop_size; ++i) {
	  
    curr = pop_in[i].phenotyp.make_state(ntors);
    current_mol = pop_in[i].mol;
    //printf("%d \n", mol);
    //	  print_molecule(mol);
    //print_state(curr);
	  
		
	
   
		
    //xyz of center of mol
    //		printf("OUT: %f \n", mol_params[j]);
		
    /*
      out[j++] = (Real) (curr->S.T.y);
      out[j++] = (Real) (curr->S.T.z);

      //quaternion wxyz
      out[j++] = (Real) (curr->S.Q.w);
      out[j++] = (Real) (curr->S.Q.x);
      out[j++] = (Real) (curr->S.Q.y);
      out[j++] = (Real) (curr->S.Q.z);
		
      for (int ii = 0; ii < MAX_ATOMS; ++ii) {
      //xyz of the atom
      out[j++] = (Real) *(curr->crd[3*ii]);
      out[j++] = (Real) *(curr->crd[3*ii +1]);
      out[j++] = (Real) *(curr->crd[3*ii +2]);
			
      //atom torsion vector xyz
      out[j++] = (Real) *(curr->vt[3*ii]);
      out[j++] = (Real) *(curr->vt[3*ii +1]);
      out[j++] = (Real) *(curr->vt[3*ii +2]);

      //atom torsion angle
      out[j++] = (Real) curr->S.tor[ii];

      //atom string
      for (unsigned int cidx = 0; cidx < MAX_CHARS; ++cidx)
      atoms[MAX_CHARS * ii + cidx] = curr->atomstr[ii][cidx];
      }
    */
  }

  //allocate global mem
  gpuErrchk(cudaMalloc ((void **) &globalReals, pop_size * MOL_INDV_SIZE));
	
  gpuErrchk(cudaMalloc ((void **) &globalChars, pop_size * MAX_ATOMS * MAX_CHARS));


  //transfer to GPU
  //	gpuErrchk(cudaMemcpy(globalReals, out, pop_size * MOL_INDV_SIZE, cudaMemcpyHostToDevice));
  //gpuErrchk(cudaMemcpy(globalChars, atoms, pop_size * MAX_ATOMS * MAX_CHARS, cudaMemcpyHostToDevice));

  //	free(out);
	
  // free(atoms);

  return true;
}
