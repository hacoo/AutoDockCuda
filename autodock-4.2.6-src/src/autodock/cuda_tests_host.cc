/* Henry Cooney <hacoo36@gmail.com> <Github: hacoo>
   AutoDockCuda: <https://github.com/hacoo/AutoDockCuda>

   cuda_tests.cu
   
   Includes tests for Cuda stuff. These tests
   happen completely on the host side, due to linking
   issues with .cu files.
   
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
#ifndef _STRUCTS_H
#include "structs.h"
#endif
#ifndef GPU_VARIABLES_H
#include "gpu_variables.h"
#endif
#ifndef QTRANSFORM
#include "qtransform.h"
#endif
#ifndef EINTCAL
#include "eintcal.h"
#endif


bool test_qtransform_kernel(Population& pop_in, int ntors,
			    CudaPtrs ptrs, double* gpu_results) {
  // Runs qtransform_kernel against CPU qtransform.
  // Compares results for each individual.
  // 
  // GPU memory should already have been allocated and transferred, 
  // and qtransform_kernel should have been run. torsion kernel
  // should NOT have run yet.
  
  int i, ii, iii;
  int pop_size = pop_in.num_individuals();
  int num_pops_to_print = pop_size;
  //int num_pops_to_print = 1;
  int state_size = 10 + ntors; // The total number of items in each STATE item
  Molecule* first_mol = pop_in[0].mol; 
  Molecule* current_mol;
  int natoms = getNumAtoms(first_mol);  
  //Real** cpu_results[pop_size];
  Real cpu_results[pop_size][MAX_ATOMS][SPACE];
  //Real cpu_results[MAX_ATOMS*SPACE];
  bool result_ok = true;
 
  State current_state;
  //print_molecule(first_mol);
  

  // Populate cpu_results array for comparison
  for(i=0; i<pop_size; ++i) {
    current_mol = pop_in[i].mol;
    current_state = pop_in[i].phenotyp.make_state(ntors);
    memcpy(cpu_results[i], current_mol->crdpdb, sizeof(double)*natoms*SPACE);
    qtransform(current_state.T, current_state.Q, cpu_results[i], natoms);
  }
  
  /*
  print_hashes();
  printf("CPU result: \n");
  printf("Pop size: %d Num atoms: %d \n", pop_size, natoms);
  for(i=0; i<num_pops_to_print; ++i) {
    printf("INDIVIDUAL: %d \n", i);
    for(ii=0; ii<natoms; ++ii) {
    printf("  %f %f %f \n", cpu_results[i][ii][0], cpu_results[i][ii][1],
	   cpu_results[i][ii][2]);
    }
    printf("\n");
  }
  print_hashes();
  
  
  print_hashes();
  printf("GPU result: \n");
  for(i=0; i<num_pops_to_print; ++i) {
    printf("INDIVIDUAL: %d \n", i);
    for(ii=0; ii<natoms; ++ii) {
      printf("  %f %f %f \n", gpu_results[i*natoms*SPACE+ii*SPACE],
	     gpu_results[i*natoms*SPACE+ii*SPACE+1], 
	     gpu_results[i*natoms*SPACE+ii*SPACE+2]);
    }
    printf("\n");
  }
  
  print_hashes();
  */

  for(i=0; i<pop_size; ++i) {
    for(ii=0; ii<natoms; ++ii) {
      for(iii=0; iii<SPACE; ++iii){
	if(cpu_results[i][ii][iii] - gpu_results[i*natoms*SPACE+ii*SPACE+iii] > 
	   0.00001) {
	  printf("ERROR in test_qtransform_kernel: Results do not agree at %d %d ",
		 i, ii);
	  if(iii == 0)
	    printf("x");
	  else if (iii == 1)
	    printf("y");
	  else
	    printf("z");
	  printf("\n");
	  result_ok = false;
	  return result_ok;
	}
      }
    }
  }
  
  return result_ok;
}



bool test_eintcal_kernel (Population& pop_in, int ntors,
			    CudaPtrs ptrs, double* gpu_results) {
  // Test the gpu eintcal kernel against CPU version
  
  int pop_size = pop_in.num_individuals();
  Molecule* first_mol = pop_in[0].mol; 
  pop_in.evaluate->compute_intermol_energy(false);
  pop_in.evaluate->compute_internal_energy(true);
  int natoms = getNumAtoms(first_mol);  
  Real* cpu_result = new Real[pop_size];
  double* gpu_result = new double[pop_size];
  
  
  for (int i=0; i<pop_size; ++i) {
    cpu_result[i] = pop_in[i].phenotyp.evaluate(Always_Eval);
  }

  

  printf("CPU Result: \n");
  for (int i=0; i<pop_size; ++i) {
    printf("  %f \n", cpu_result[i]);
  }

  printf("GPU Result: \n");
  for (int i=0; i<pop_size; ++i) {
    printf("  %f \n", gpu_result[i]);
  }
  
  
  

  
  delete cpu_result;
  return false;
  
}
