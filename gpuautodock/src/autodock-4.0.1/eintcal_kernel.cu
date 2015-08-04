/*
  Henry Cooney <hacoo36@gmail.com> <https://github.com/hacoo>

  eintcal_kernel.cu
  AutoDockCuda <https://github.com/hacoo/AutoDockCuda>

  Improved internal energy calculation kernel for AutoDockCuda.
  Calculates the total internal energy of the ligand, and the 
  internal energy of the receptor if it is flexible. Returns

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

#include "eintcal_kernel.h"

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



/**
 * eintcal GPU kernel, does eintcal energy calculations for each 
 * individual in the population.
 * @param num_individualsgpu number of individuals in population
 * @param natomsgpu number of atoms
 * @param penergiesgpu array of energies used to store individual's energy
 * @param nonbondlist (used in cpu eintcal)
 * @param tcoord (used in cpu eintcal)
 * @param B_include_1_4_interactions (used in cpu eintcal)
 * @param B_have_flexible_residues (used in cpu eintcal)
 * @param nnb_array (used in cpu eintcal)
 * @param total_nonbond_number - total number of nonbond atoms
 * @param Nb_group_energy (used in cpu eintcal)
 * @param stre_vdW_Hb (used in cpu eintcal)
 * @param strsol_fn (used in cpu eintcal)
 * @param strepsilon_fn (used in cpu eintcal)
 * @param strr_epsilon_fn (used in cpu eintcal)
 * @param b_comp_intermolgpu (used in cpu eintcal)
 * @param pfloat_arraygpu array of float variables used in cpu trilinterp
 * @param pint_arraygpu array of integer varibales used in cpu trilinterp
 */

__global__ void eintcal_kernel_per_block(
                        unsigned int num_individualsgpu,
                        int natomsgpu, 
                        float *penergiesgpu, 
                        float *nonbondlist, 
                        float *tcoord, 
                        int B_include_1_4_interactions, 
                        int B_have_flexible_residues, 
                        int *nnb_array, 
			int total_nonbond_number,
                        float *Nb_group_energy, 
                        float *stre_vdW_Hb, 
                        float *strsol_fn, 
                        float *strepsilon_fn, 
                        float *strr_epsilon_fn,
                        int b_comp_intermolgpu,
                        float *pfloat_arraygpu,
                        int *pint_arraygpu)
{
  extern __shared__ float total_internal_energies[]; // This will contain each atom's energy. Then,
  // we will do a list reductiion to get total energy.
  int idx = blockIdx.x;
  float dx = 0.0f, dy = 0.0f, dz = 0.0f;
  float r2 = 0.0f;
    
  float total_e_internal = 0.0f;
  float e_elec = 0.0f;
  int inb = threadIdx.x;
 
  int a1 = (int)nonbondlist[inb * 7 + 0];
  int a2 = (int)nonbondlist[inb * 7 + 1];
  int t1 = (int)nonbondlist[inb * 7 + 2];
  int t2 = (int)nonbondlist[inb * 7 + 3];

  int nonbond_type = (int)nonbondlist[inb * 7 + 4];
  float nb_desolv = nonbondlist[inb * 7  + 5];
  float q1q2 = nonbondlist[inb * 7 + 6];

  int index_1t_NEINT = 0;
  int index_1t_NDIEL = 0;
  int nb_group = 0;
	

  if (idx < total_nonbond_number)
    {
   
      if (!pint_arraygpu[INTEVALFLAG * num_individualsgpu + idx])
        {
    
#ifndef EINTCALPRINT
#   ifndef NOSQRT
	  float r = 0.0f;
	  float nbc = (Boole)pint_arraygpu[INTNONBONDCUT * num_individualsgpu + idx] ? NBC : 999;
#   else
	  float nbc2 = (Boole)pint_arraygpu[INTNONBONDCUT * num_individualsgpu + idx] ? NBC2 : 999 * 999;
#   endif

#else
#   ifndef NOSQRT
	  float d = 0.0f;
	  float nbc = (Boole)pint_arraygpu[INTNONBONDCUT * num_individualsgpu + idx] ? NBC : 999;
#   else
	  float nbc2 = (Boole)pint_arraygpu[INTNONBONDCUT * num_individualsgpu + idx] ? NBC2 : 999 * 999;
#   endif
#endif

	  // By default, we have one nonbond group, (1) intramolecular in the ligand
	  // If we have flexible residues, we need to consider three groups of nonbonds:
	  // (1) intramolecular in the ligand, (2) intermolecular and (3) intramolecular in the receptor

	  int nb_group_max = B_have_flexible_residues? 3 : 1 ;

	  for (nb_group = 0; nb_group < nb_group_max; nb_group++)
	    {                
	      // Each thread will handle either one atom (if receptor is nonflexible)
	      // or 3 atoms (if receptor is flexible). This could be further parallelized
	      // later.

	      float e_internal = 0.0f;
	      float e_desolv = 0.0f;

	      dx = tcoord[idx * natomsgpu * SPACE + a1 * SPACE + X] - tcoord[idx * natomsgpu * SPACE + a2 * SPACE + X];
	      dy = tcoord[idx * natomsgpu * SPACE + a1 * SPACE + Y] - tcoord[idx * natomsgpu * SPACE + a2 * SPACE + Y];
	      dz = tcoord[idx * natomsgpu * SPACE + a1 * SPACE + Z] - tcoord[idx * natomsgpu * SPACE + a2 * SPACE + Z];

#ifndef NOSQRT
	      r = clamp(hypotenuse(dx,dy,dz), RMIN_ELEC);
	      r2 = r*r;
	      int index = Ang_to_index(r);

#else
	      r2 = sqhypotenuse(dx,dy,dz);
	      r2 = clamp(r2, RMIN_ELEC2);
	      int index = SqAng_to_index(r2);
#endif

	      index_1t_NEINT = BoundedNeint(index);
	      index_1t_NDIEL = BoundedNdiel(index);
	      
	      if ((Boole)pint_arraygpu[INTINCELEC * num_individualsgpu + idx])
		{
		  float r_dielectric = strr_epsilon_fn[index_1t_NDIEL];
		  e_elec = q1q2 * r_dielectric;
		  e_internal = e_elec;
		}
                   
	      if (r2 < nbc2)
		{
		  e_desolv = strsol_fn[index_1t_NEINT] * nb_desolv;
		  int myidx;
		  if (B_include_1_4_interactions != 0 && nonbond_type == 4)
		    {
		      myidx = index_1t_NEINT * ATOM_MAPS * ATOM_MAPS + t2 * ATOM_MAPS + t1;
		      if (myidx == NEINT * ATOM_MAPS * ATOM_MAPS)
			{
			  e_internal += pfloat_arraygpu[FLOATSCALE14 * num_individualsgpu + idx] * (stre_vdW_Hb[myidx-1] + e_desolv);
			}
		      else
			{
			  e_internal += pfloat_arraygpu[FLOATSCALE14 * num_individualsgpu + idx] * (stre_vdW_Hb[myidx] + e_desolv);
			}
		    } else {
		    myidx = index_1t_NEINT * ATOM_MAPS * ATOM_MAPS + t2 * ATOM_MAPS + t1;
		    if (myidx == NEINT * ATOM_MAPS * ATOM_MAPS)
		      {
			e_internal += stre_vdW_Hb[myidx-1] + e_desolv;

		      }
		    else
		      {
			e_internal += stre_vdW_Hb[myidx] + e_desolv;
		      }

		  }
		}
	      total_internal_energies[inb] = e_internal;
	    }

	  __syncthreads();
	  // Now do a list reduction on total_internal_energies. There's no need to use a 
	  // work-efficient scan here, since we've already recruited numatoms threads. Naive scan is ok

	  // For now, just cheat an have thread 0 do it.

	  if(threadIdx.x == 0) {
	    int i;
	    for(i=0; i<total_nonbond_number-1; ++i){
	      total_internal_energies[total_nonbond_number-1] += total_internal_energies[i];
	    }
	  }

	  __syncthreads();

	  

	  

	  if (nb_group == INTRA_LIGAND) 
	    {
	      Nb_group_energy[INTRA_LIGAND] = total_e_internal;
	    } else if (nb_group == INTER) {
	    Nb_group_energy[INTER] = total_e_internal - Nb_group_energy[INTRA_LIGAND];
	  } else if (nb_group == INTRA_RECEPTOR) {
	    Nb_group_energy[INTRA_RECEPTOR] = total_e_internal - Nb_group_energy[INTRA_LIGAND] - Nb_group_energy[INTER];
	  }

	}

      if(b_comp_intermolgpu)
	{
	  penergiesgpu[idx] += ((float)total_e_internal - pfloat_arraygpu[FLOATUNBOUNDINTERNAL * num_individualsgpu + idx]);
	}
      else
	{
	  penergiesgpu[idx] = ((float)total_e_internal - pfloat_arraygpu[FLOATUNBOUNDINTERNAL * num_individualsgpu + idx]);
	}
	    
    }
}


  
  
  
 

