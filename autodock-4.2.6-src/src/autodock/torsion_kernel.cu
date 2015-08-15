#include "constants.h"
#ifndef CUDA_HEADERS
#include "cuda.h"
#include "cuda_runtime.h"
#endif
#ifndef CUDA_STRUCTS_H
#include "cuda_structs.h"
#endif

__global__ void torsion_kernel(CudaPtrs ptrs) {

//void torsion( const State& now, //tors @ states[i*statesize+10]
//    /* not const */ Real crd[MAX_ATOMS][SPACE], //atom_crds
//              const Real v[MAX_TORS][SPACE], //torsions_dev 
//              const int tlist[MAX_TORS][MAX_ATOMS], //torsion_root_list
//              const int ntor ) //ptrs->ntors_dev
    register double crdtemp[SPACE]; //shared
    register double  d[SPACE];
    register double  k[SPACE][SPACE]; //shared
    register double s, c, o;          /* "o" is: 1. - c, "One minus c". */
    int mvatm, numatmmoved;
    //double twopi=6.28318530717958647692;

    int state_size = *ptrs.state_size_dev;
    int ix = threadIdx.x;
    int iy = threadIdx.y;
    int n = blockIdx.x;

    if(ix == 0 && iy == 0)
       s = sin(fmod(ptrs.states_dev[n*state_size+10], (double)TWOPI));
    if(ix == 1 && iy == 0)
       c = cos(fmod(ptrs.states_dev[n*state_size+10], (double)TWOPI));
    if(ix == 2 && iy == 0)
       o = 1. - cos(fmod(ptrs.states_dev[n*state_size+10], (double)TWOPI));

    __syncthreads();

    if(ix == 0)
    {
        int temp = ptrs.torsion_root_list_dev[n];
        crdtemp[iy] = (double) ptrs.atom_crds_dev[temp*(iy+1)];
    }
    else if (ix == 1)
        k[iy][iy] = o * ptrs.torsions_dev[n*(iy+1)] * ptrs.torsions_dev[n*(iy+1)] + c;
    else if ((ix < 6) && (ix != iy))
        k[iy][ix] = ptrs.torsions_dev[n*(iy+1)] * (o * ptrs.torsions_dev[n*(ix+1)]) - (s * ptrs.torsions_dev[n*(iy - ix - 3 + 1)]);
    else if ((ix == 6) && (iy == 0))
        numatmmoved = ptrs.torsion_root_list_dev[n*(NUM_ATM_MOVED+1)] + 3;

    __syncthreads();

    if(ix <= numatmmoved)
    {
        mvatm = ptrs.torsion_root_list_dev[n*(ix+1)];
        d[iy] = (double)ptrs.atom_crds_dev[mvatm*(iy+1)] - crdtemp[iy];
        __syncthreads();
        ptrs.atom_crds_dev[mvatm*(iy+1)] = (double)crdtemp[iy] + d[0] * k[iy][0] + d[1] * k[iy][1] + d[2] * k[iy][2];
    }
}
