

/* Henry Cooney <hacoo36@gmail.com> <Github: hacoo>
   AutoDockCuda: <https://github.com/hacoo/AutoDockCuda>

   cudat_utils_host.h
   
   Includes utility functions for dealing with CUDA, on the host side.
   Also includes print functions for debugging and examining data.

*/

#ifndef CUDA_UTILS_HOST_H
#define CUDA_UTILS_HOST_H
#endif
#ifndef CUDA_HEADERS
#include "/pkgs/nvidia-cuda/5.5/include/cuda.h"
#include "/pkgs/nvidia-cuda/5.5/include/cuda_runtime.h"
#endif
#ifndef _STRUCTS_H
#include "structs.h"
#endif



#define gpuErrchk(ans) { gpuAssert((ans), __FILE__, __LINE__); }
inline void gpuAssert(cudaError_t code, const char *file, int line, bool abort=true)
{
  if (code != cudaSuccess) 
    {
      fprintf(stderr,"GPUassert: %s %s %d\n", cudaGetErrorString(code), file, line);
      if (abort) exit(code);
    }
}


void print_quat(Quat q);
void print_energy(Energy e);
void print_coord(Coord c);
void print_state(State s);
void print_molecule(Molecule* m);
int getNumAtoms(Molecule* m);
char* getAtomString(Molecule* m, int n);
char* getAtomStringArray(Molecule* m);
void freeAtomStringArray(char** a, int numatoms);
double* getTorsions(Molecule* m, int ntors);
double* getAtomCrds(Molecule* m);
int* getTorsionRootList(Molecule* m, int ntors);
void print_double_matrix(double* arr, int width, int height);
void print_hashes();
