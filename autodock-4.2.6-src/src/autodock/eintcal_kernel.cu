/* Henry Cooney <hacoo36@gmail.com> <Github: hacoo>
   AutoDockCuda: <https://github.com/hacoo/AutoDockCuda>
   
   eintcal_kernel.cu

   CUDAfied version of eintcal. eintcal does internal energy calculation
   for a ligand.

   Internal energy is the energy of the ligand due to the configuration of 
   the ligand (i.e. how it is bent or flexed). 
   
   Each atom in the ligand contributes to the overall internal energy.
   Additionally, if the receptor is flexible, the receptor's internal
   energy will be calculated as well (not implemented yet).
*/

#include "constants.h"

// Memory allocation functions -- move to memory_layout.cu





__global__
void eintcal_kernel() {
  // Does internal energy calculation for a population array
  // which is already on the GPU.
  
  
}


