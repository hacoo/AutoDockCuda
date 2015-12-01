/* Henry Cooney <hacoo36@gmail.com> <Github: hacoo>
   AutoDockCuda: <https://github.com/hacoo/AutoDockCuda>

   cuda_tests.cu
   
   Includes tests for Cuda stuff.
   
*/

#include "constants.h"
#include "typedefs.h"
#include <string.h>
#include <stdio.h>
#include <stdlib.h>
#ifndef _SUPPORT_H
#include "support.h"
#endif
#ifndef CUDA_UTILS_HOST_H
#include "cuda_utils_host.h"
#endif
#ifndef _AUTOCOMM
#include "autocomm.h"
#endif
#ifndef CUDA_STRUCTS_H
#include "cuda_structs.h"
#endif
#ifndef CUDA_HEADERS
#include "/pkgs/nvidia-cuda/5.5/include/cuda.h"
#include "/pkgs/nvidia-cuda/5.5/include/cuda_runtime.h"
#endif
#ifndef _STRUCTS_H
#include "structs.h"
#endif
#ifndef GPU_VARIABLES_H
#include "gpu_variables.h"
#endif



bool test_memory_transfer(Population& pop_in, int ntors, CudaPtrs* ptrs) {
  // Tests that memory already transfered to the GPUb
  // can be transfered back in one piece.
  // Right now correct transfer is verified by inspection -- will
  // need to automate this later
  
  int i, ii;
  Molecule* m = pop_in[0].mol;
  int natoms = getNumAtoms(m);
  int pop_size = pop_in.num_individuals();
  int state_size = 10 + ntors; // The total number of items in each STATE item
  // - 3 trans coords + 4 quat coords + 3 center coords + ntors torsions

  
  int natoms_t, ntors_t, state_size_t;
  double* atom_crds_t = (double*) malloc(sizeof(double)*natoms*SPACE);
  char* atom_strings_t = (char*) malloc(sizeof(char)*natoms*MAX_CHARS);
  double* torsions_t = (double*) malloc(sizeof(double)*ntors*SPACE);
  int* torsion_root_list_t = (int*) malloc(sizeof(int)*natoms*ntors);
  double* states_t  = (double*) malloc(sizeof(double)*state_size*pop_size);
  
  gpuErrchk(cudaMemcpy(torsion_root_list_t, ptrs->torsion_root_list_dev,  
		       sizeof(int)*natoms*ntors, cudaMemcpyDeviceToHost));

  for(i=0; i<natoms; ++i){
    gpuErrchk(cudaMemcpy(atom_strings_t+i*MAX_CHARS, ptrs->atom_strings_dev+i*MAX_CHARS,
	      MAX_CHARS,
	      cudaMemcpyDeviceToHost));
  }

  gpuErrchk(cudaMemcpy(atom_crds_t, ptrs->atom_crds_dev,  
		       sizeof(double)*natoms*SPACE, cudaMemcpyDeviceToHost));
  
  gpuErrchk(cudaMemcpy(torsions_t, ptrs->torsions_dev,  
		       sizeof(double)*ntors*SPACE, cudaMemcpyDeviceToHost));

  gpuErrchk(cudaMemcpy(&natoms_t, ptrs->natoms_dev,  
		       sizeof(int), cudaMemcpyDeviceToHost));
  gpuErrchk(cudaMemcpy(&ntors_t, ptrs->ntors_dev,  
		       sizeof(int), cudaMemcpyDeviceToHost));
  gpuErrchk(cudaMemcpy(&state_size_t, ptrs->state_size_dev,  
		       sizeof(int), cudaMemcpyDeviceToHost));

  
  
  printf("Contents of atom string array: \n");
  for (i=0; i<natoms; ++i) {
    printf("  %s\n", atom_strings_t+i*MAX_CHARS);
  }
  
  printf("Root List: \n");
  for (i=0; i<ntors; ++i) {
    for (ii=0; ii<natoms; ++ii) {
      printf("%d ", torsion_root_list_t[i*natoms + ii]);
    }
    printf("\n");
  }

  printf("Contents of atom_crds: \n");
  for (i=0; i<natoms; ++i) {
    printf(" %f %f %f \n", atom_crds_t[3*i], atom_crds_t[3*i+1], atom_crds_t[3*i+2]);
  }

  printf("Torsions: \n");
  for (i=0; i<ntors; ++i) {
  printf("  %f %f %f \n", torsions_t[3*i], torsions_t[3*i+1], torsions_t[3*i+2]);
  }

  printf("There are %d atoms. \n", natoms_t);
  printf("There are %d torsions. \n", ntors_t);
  printf("The state size is %d. \n", state_size_t);
  
  gpuErrchk(cudaMemcpy(states_t, ptrs->states_dev,
		       sizeof(double)*pop_size*state_size, cudaMemcpyDeviceToHost));
  
  for(int i=0; i<pop_size; ++i) {
    printf("STATE: %d \n", i);
    
    printf("T:   %f %f %f \n", states_t[i*state_size],
	   states_t[i*state_size+1], states_t[i*state_size+2]);
    printf("Q:   %f %f %f %f\n", states_t[i*state_size+3],
	   states_t[i*state_size+4], states_t[i*state_size+5], states_t[i*state_size+6]);
    printf("C:   %f %f %f \n", states_t[i*state_size+7],
	   states_t[i*state_size+8], states_t[i*state_size+9]);
    printf("Tors: ");
    for(ii=0; ii<ntors; ++ii){
      printf("%f ", states_t[i*state_size+10+ii]);
    }
    printf("\n");
  }

  // Test eintcal memory transfer:
  
  
  //free(atom_crds_t);
  free(atom_strings_t);
  free(torsions_t);
  free(torsion_root_list_t);
  free(states_t);

  return false;
}




