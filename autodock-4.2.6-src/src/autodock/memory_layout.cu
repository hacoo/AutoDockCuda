#include "constants.h"
#include "typedefs.h"
#include "/pkgs/nvidia-cuda/5.5/include/cuda.h"
#include "/pkgs/nvidia-cuda/5.5/include/cuda_runtime.h"
#include <stdio.h>
#ifndef _SUPPORT_H
#include "support.h"
#endif


const int ATOM_SIZE = (6 + MAX_TORS) * 3 * sizeof(Real);
const int MOL_INDV_SIZE = (7 + MAX_TORS) * sizeof(Real) + MAX_ATOMS * ATOM_SIZE;

Real * globalReals;
char * globalChars;

enum ATTRIBUTE {
	xyz = 0,
	wxyz = 3
};

/////////////////////***********************************************/////////////////////
///****   THESE ARE THE UTILITY FUNCTIONS TO USE TO ACCESS DATA ON THE GPU    *****/////

__device__ Real * getIndvAttribute(int idx, ATTRIBUTE a) {
	//all data is packed into array in x,y,z,qw,qx,qy,qz, [torsion data], ......
	//returns the start address, move to next item by adding sizeof(Real)
	return globalReals + (idx * MOL_INDV_SIZE + a) * sizeof(Real);
}

__device__ Real * getTorsion(int indvIdx, int torsionIdx) {
	//all data is packed into array in x1,y1,z1,theta1, x2,y2, .....
	//returns the start address, move to next item by adding sizeof(Real)
	return globalReals + (indvIdx * MOL_INDV_SIZE + 7 + 4 * torsionIdx) * sizeof(Real);
}

__device__ char * getAtom(int indvIdx, int atom) {
	//all data is packed into array in c11,c12,...c1MAX_CHARS, c21, c22, c23, ...
	//returns the start address, move to next item by adding sizeof(char)
	return globalReals + (idx * MAX_TORSIONS * MAX_CHARS + atom * MAX_CHARS) * sizeof(char);
}

/////////////////////// ^^^^^utility ^^^^^^^ //////////////////////////////
/////////////////////////////////////////////////////////////////////////


// this function allocates memory on the gpu in form of compact arrays
// called globalReals and globalChars. Then it converts a population to array form
// and transfers all the data to the gpu at once

bool allocate_pop_to_gpu(Population & pop_in) {
	//allocates several arrays of various types to be moved to the GPU

	Real * out; // this contains most of the data
	char * atoms; // this contains the atom data
	bool succ;

	cudaError succ;

	int pop_size = pop_in.num_individuals();

	succ = cudaMalloc((void **) &out, pop_size * MOL_INDV_SIZE);
	if (cudaSuccess != succ)
		return false;
	succ = cudaMalloc((void **) &atoms, pop_size * MAX_ATOMS * MAX_CHARS);
	if (cudaSuccess != succ)
		return false;

	for (int i = 0; i < pop_size; ++i) {
		if (pop_in[i].mol == NULL) {
			printf("no molecule for individual %d", i);
			return false;
		}

		Molecule * curr = pop_in[i].mol;
	
		int j = MOL_INDV_SIZE * i; //output idx
		
		//xyz of center of mol
		out[j++] = (Real) (curr->S.T.x);
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

	}

	//allocate global mem
	succ = cudaMalloc ((void **) &globalReals, pop_size * MOL_INDV_SIZE);
	if (cudaSuccess != succ)
		return false;

	succ = cudaMalloc ((void **) &globalChars, pop_size * MAX_ATOMS * MAX_CHARS);
	if (cudaSuccess != succ)
		return false;

	//transfer to GPU
	succ = cudaMemcpy(globalReals, out, pop_size * MOL_INDV_SIZE, cudaMemcpyHostToDevice);
	if (cudaSuccess != succ)
		return false;
	succ = cudaMemcpy(globalChars, atoms, pop_size * MAX_ATOMS * MAX_CHARS, cudaMemcpyHostToDevice);
	if (cudaSuccess != succ)
		return false;

	succ = cudaFree(out);
	if (cudaSuccess != succ)
		return false;
	
	succ = cudaFree(atoms);
	if (cudaSuccess != succ)
		return false;
	
	
	
	return true;
}
