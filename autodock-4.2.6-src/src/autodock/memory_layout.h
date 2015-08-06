/* Henry Cooney <hacoo36@gmail.com> <Github: hacoo>
   AutoDockCuda: <https://github.com/hacoo/AutoDockCuda>

   memory_layout.h
    
   Header file for memory_layout.cu, written by Patrick Romero <Github: patrick38894>
   
   memory_layout.cu includes functions for initializing a population on
   the GPU.

*/


#ifndef MEMORY_LAYOUT_H
#define MEMORY_LAYOUT_H
#endif

const int ATOM_SIZE = (6 + MAX_TORS) * 3 * sizeof(Real);
const int MOL_INDV_SIZE = (7 + MAX_TORS) * sizeof(Real) + MAX_ATOMS * ATOM_SIZE;

__global__ Real * globalReals;
__global__ char * globalChars;


__constant__ enum ATTRIBUTE {
	xyz = 0,
	wxyz = 3
};


// Function prototypes
bool allocate_pop_to_gpu(Population & pop_in);
//__global__ Real * getIndvAttribute(int idx, ATTRIBUTE a);
//__global__ Real * getTorsion(int indvIdx, int torsionIdx);
//__global__ Real * getTorsion(int indvIdx, int torsionIdx);




