/* Henry Cooney <hacoo36@gmail.com> <Github: hacoo>
   AutoDockCuda: <https://github.com/hacoo/AutoDockCuda>

   cuda_tests.cu
   
   Includes tests for Cuda stuff. These tests
   happen completely on the host side, due to linking
   issues with .cu files.
   
*/


#ifndef CUDA_TESTS_HOST_H
#define CUDA_TESTS_HOST_H
#endif
#ifndef CUDA_STRUCTS_H
#include "cuda_structs.h"
#endif


bool test_qtransform_kernel(Population& pop_in, int ntors,
			      CudaPtrs ptrs, double* gpu_results);

