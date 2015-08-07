
/* Henry Cooney <hacoo36@gmail.com> <Github: hacoo>
   AutoDockCuda: <https://github.com/hacoo/AutoDockCuda>

   start_CUDA_on_population.cc

   Creates an interface between the Population class
   and CUDA evaluation of the Population. Functions
   in this file will kick off CUDA evaluation, and 
   return an array of individual fitness values computed 
   on the GPU.
*/

#ifndef _SUPPORT.h
#include "support.h"
#endif
#include <assert.h>
#ifndef MEMORY_LAYOUT_H
#include "memory_layout.cuh"
#endif

double* Population::evaluate_on_GPU() {
  // Evalueates each individual of the population 
  // in parallel on the GPU. Will return an array of 'values',
  // each value is the fitness value of the corresponding 
  // individual. The number of values returned is equal to the 
  // number of individuals in the population (i.e. Population.size)

  bool GPU_pop_allocate_success = allocate_pop_to_gpu(*this);
  assert(GPU_pop_allocate_success == true);


  return NULL;
}



