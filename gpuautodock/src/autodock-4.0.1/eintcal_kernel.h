/*
  Henry Cooney <hacoo36@gmail.com> <https://github.com/hacoo>

  eintcal_kernel.h
  AutoDockCuda <https://github.com/hacoo/AutoDockCuda>

  Improved internal energy calculation kernel for AutoDockCuda.
  Calculates the total internal energy of the ligand, and the 
  internal energy of the receptor if it is flexible.

  This kernel uses a per-block approach to acheive good performance.
  Autodock usses a Lamarckian Genetic Algorithm (LGA) to find the ligand's
  docked configuration, each possible configuration is an 'individual'. 
  In this kernel, each Block contains a single individual which is 
  loaded completely into shared memory.

  When the kernel is completed, the resulting free energy is loaded back to
  global memory.
  
  This kernel is intended to be used with Autodock 4.0.1
  
  It contains code from the Autodock 4.2.6 (non-CUDA) source and from
  gpuautodock (http://sourceforge.net/projects/gpuautodock/)
  
  Thank you to Sarnath Kannan, whose paper (http://www0.cs.ucl.ac.uk/staff/ucacbbl/cigpu2010/papers/c-7216.pdf) describes an efficient CUDA implementation
  of Autodock.

*/


// Need #include guards?
#include <cuda.h>
#include <cuda_runtime.h>
#include <stdio.h>
#include "typedefs.h"

#include "autocomm.h"
#include "grid.h"
#include "eval.h"
#include "constants.h"
#include "trilinterp.h"
#include "eintcal.h"
#include "distdepdiel.h"
#include "cuda_wrapper.h"


__global__ void eintcal_kernel_per_block(
                        unsigned int num_individualsgpu,
                        int natomsgpu, 
                        float *penergiesgpu, 
                        float *nonbondlist, 
                        float *tcoord, 
                        Boole B_include_1_4_interactions, 
                        Boole B_have_flexible_residues, 
                        int *nnb_array, 
                        float *Nb_group_energy, 
                        float *stre_vdW_Hb, 
                        float *strsol_fn, 
                        float *strepsilon_fn, 
                        float *strr_epsilon_fn,
                        Boole b_comp_intermolgpu,
                        float *pfloat_arraygpu,
                        int *pint_arraygpu);

