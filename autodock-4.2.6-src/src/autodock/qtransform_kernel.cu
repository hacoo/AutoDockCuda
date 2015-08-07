/* Henry Cooney <hacoo36@gmail.com> <Github: hacoo>
   AutoDockCuda: <https://github.com/hacoo/AutoDockCuda>
   
   qtransform_kernel.cu
   
   CUDA implementation of qtransform. qtransform is responsible for 
   applying rotation and translation to all atoms in a ligand.

   This kernel should be run on a population array already
   stored on the GPU. After it is run, the coordinates of each
   atom in each individual will now be updated to reflect
   their translation and quaternion rotation.

*/


// Initial kernel will handle just one atom for testing...
