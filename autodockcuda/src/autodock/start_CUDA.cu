
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
#ifndef EINTCAL_KERNEL_H
#include "eintcal_kernel.cuh"
#endif
#ifndef TORSION_KERNEL_H
#include "torsion_kernel.cuh"
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

  // Torsion kernel is torsion per block
  printf("Launching torsion_kernel with dimGrid: %d and dimBlock: %d,3,3 \n", pop_size, ntors);
  torsion_kernel<<<dim3(pop_size), dim3(ntors,3,3)>>>(ptrs);
  cudaError_t err1 = cudaGetLastError();
  if(err1 != cudaSuccess){
      printf("Error after torsion_kernel: %s\n", cudaGetErrorString(err1));
      cudaErrorDetected = 1;
  }
  double* tor_results = (double*) malloc(sizeof(double)*natoms*SPACE);
  gpuErrchk(cudaMemcpy(tor_results, ptrs.indiv_crds_dev, 
                sizeof(double)*natoms*SPACE, cudaMemcpyDeviceToHost));

  // qtransform kernel launch
  printf("Lanching qtransform_kernel with dimGrid: %d dimBlock: %d, for %d atoms \n",
	 pop_size, natoms, natoms);
  qtransform_kernel<<<dimGrid, dimBlock>>>(ptrs);
  cudaError_t err = cudaGetLastError();
  if (err != cudaSuccess) {
    printf("Error after qtransform_kernel: %s\n", cudaGetErrorString(err));
    cudaErrorDetected = 1;
  }

  // Eintcal is per nonbond -- not per atom. So, use a block size of 1024,
  // and request as many blocks as necessary.
  int Nnb = this_pop->evaluate->get_Nnb();
  int num_eintcal_blocks = (Nnb/512+ 1);
  printf("Lanching eintcal_kernel with dimGrid: %d,%d dimBlock: 512, for %d nonbonds \n",
	 num_eintcal_blocks, pop_size, Nnb);
  eintcal_kernel<<<dim3(num_eintcal_blocks, pop_size), dim3(512, 1, 1)>>>(ptrs);
  err = cudaGetLastError();
  if (err != cudaSuccess) {
    printf("Error after eintcal_kernel: %s\n", cudaGetErrorString(err));
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
    printf("ERROR: test_qtransform_kernel -- FAILED \n");
  else
    printf("test_qtransform_kernel -- OK \n");
  free(quat_results);
 
  // Test Torsion
  if (!test_torsion_kernel(*this_pop, ntors, ptrs, tor_results))
    printf("ERROR: test_torsion_kernel -- FAILED \n");
  else
    printf("test_torsion_kernel -- OK \n");
  free(tor_results);


  // Test eintcal
  double* eintcal_results = (double*) malloc(sizeof(double)*pop_size);
  gpuErrchk(cudaMemcpy(eintcal_results, ptrs.internal_energies_dev, 
		       sizeof(double)*pop_size, cudaMemcpyDeviceToHost));

  if (!test_eintcal_kernel(*this_pop, ntors, ptrs, eintcal_results))
      printf("ERROR: test_eintcal_kernel -- FAILED \n");
  else
    printf("test_eintcal_kernel -- OK \n");

  free(eintcal_results);
      
  

  //////////////////////////
  
  
  printf("Done! \n");  
  if(cudaErrorDetected) {
    printf("!!!!!!!!! WARNING !!!!!!!!!\n");
    printf("CUDA ERROR DETECTED -- CHECK OUTPUT \n");
    printf("!!!!!!!!!!!!!!!!!!!!!!!!!!!\n");
  }
}

