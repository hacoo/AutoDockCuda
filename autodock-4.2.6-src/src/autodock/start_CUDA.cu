
/* Henry Cooney <hacoo36@gmail.com> <Github: hacoo>
   AutoDockCuda: <https://github.com/hacoo/AutoDockCuda>

   start_CUDA.cu
   
   This file's job is to start the actual CUDA kernel. It
   should call needed kernels in order.

   This file also defines gpu memory pointers. Because CUDA
   is picky about how files are #included together, you should 
   #include all kernels after the gpu memory pointer decs.
   
*/

#ifndef _SUPPORT_H
#include "support.h"
#endif
#ifndef CUDA_HEADERS
#include "/pkgs/nvidia-cuda/5.5/include/cuda.h"
#include "/pkgs/nvidia-cuda/5.5/include/cuda_runtime.h"
#endif
#include "cuda_utils_host.h"
#include "memory_layout.cuh"
#include "cuda_utils_gpu.cuh"
#ifndef CUDA_STRUCTS_H
#include "cuda_structs.h"
#endif
#ifndef CUDA_TESTS_H
#include "cuda_tests.cuh"
#endif
#ifndef CUDA_TESTS_HOST_H
#include "cuda_tests_host.h"
#endif
#ifndef QTRANSFORM_KERNEL_H
#include "qtransform_kernel.cuh"
#endif





void start_CUDA_on_population(Population* this_pop, int ntors) {
  // Begins evaluation of Population on GPU. For now, this is a 
  // placeholder that will allow kernels to be tested, etc.
  
  CudaPtrs ptrs;
  int cudaErrorDetected = 0;
  int natoms = getNumAtoms((*this_pop)[0].mol);
  int pop_size = this_pop->num_individuals();

  printf("Allocating %d atoms and %d torsions to GPU...\n", natoms, ntors);
  allocate_pop_to_gpu(*this_pop, ntors, &ptrs);
  
  ////// RUN CUDA KERNELS //////
  printf("Starting kernels. Block size: %d Grid size: %d\n", natoms, pop_size);
  dim3 dimBlock(natoms);
  dim3 dimGrid(pop_size);

  qtransform_kernel<<<dimGrid, dimBlock>>>(ptrs);
  cudaError_t err = cudaGetLastError();
  if (err != cudaSuccess) {
    printf("Error: %s\n", cudaGetErrorString(err));
    cudaErrorDetected = 1;
  }


  //////////////////////////////

  ////// TEST SECTION //////
  printf("Now running tests. \n");

  // Test memory transfer
  //test_memory_transfer(*this_pop, ntors, &ptrs);
  
  // Test qtransform
  double* quat_results = (double*) malloc(sizeof(double)*natoms*pop_size*SPACE);
  gpuErrchk(cudaMemcpy(quat_results, ptrs.indiv_crds_dev, 
		       sizeof(double)*pop_size*natoms*SPACE, cudaMemcpyDeviceToHost));

  if (!test_qtransform_kernel(*this_pop, ntors, ptrs, quat_results))
    printf("ERROR: test_qtransform_kernel failed \n");
  free(quat_results);

  // Test eintcal
  double* eintcal_results = (double*) malloc(sizeof(double)*pop_size);
  if (!test_eintcal_kernel(*this_pop, ntors, ptrs, eintcal_results))
      printf("ERROR: test_eintcal_kernel failed \n");
  free(eintcal_results);
      
  

  //////////////////////////
  
  
  printf("Done! \n");  
  if(cudaErrorDetected) {
    printf("!!!!!!!!! WARNING !!!!!!!!!\n");
    printf("CUDA ERROR DETECTED -- CHECK OUTPUT \n");
    printf("!!!!!!!!!!!!!!!!!!!!!!!!!!!\n");
  }
}

