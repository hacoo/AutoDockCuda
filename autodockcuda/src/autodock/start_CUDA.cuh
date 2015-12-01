
/* Henry Cooney <hacoo36@gmail.com> <Github: hacoo>
   AutoDockCuda: <https://github.com/hacoo/AutoDockCuda>

   start_CUDA.cuh
   
   This file's job is to start the actual CUDA kernel. It
   should call needed kernels in order.

   This file also defines gpu memory pointers. Because CUDA
   is picky about how files are #included together, you should 
   #include all kernels after the gpu memory pointer decs.
   
*/

#ifndef START_CUDA_H
#define START_CUDA_H
#endif

void start_CUDA_on_population(Population* this_pop, int ntors);
