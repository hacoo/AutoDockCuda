
/* Henry Cooney <hacoo36@gmail.com> <Github: hacoo>
   AutoDockCuda: <https://github.com/hacoo/AutoDockCuda>

   gpu_variables.cuh
   
   contains GPU variable definitions.

*/

#ifndef GPU_VARIABLE_H
// GPU pointers:
__constant__ double atom_crds_dev[MAX_ATOMS*SPACE];
__constant__ int* natoms_dev;
__constant__ double torsions_dev[MAX_TORS*SPACE]; 
extern int torsion_root_list_dev[MAX_ATOMS*MAX_TORS];
extern char* atom_strings_dev[MAX_ATOMS];

#define GPU_VARIABLE_H
#endif
