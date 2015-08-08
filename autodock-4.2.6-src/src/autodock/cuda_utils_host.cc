
/* Henry Cooney <hacoo36@gmail.com> <Github: hacoo>
   AutoDockCuda: <https://github.com/hacoo/AutoDockCuda>

   cudat_utils_host.h
   
   Includes utility functions for dealing with CUDA, on the host side.
   Also includes print functions for debugging and examining data.

*/

#ifndef _STRUCTS_H
#include "structs.h"
#endif
#include "constants.h"

void print_quat(Quat q) {
  printf("Quaternion: x: %f, y: %f, z: %f, w: %f \n", q.x, q.y, q.z, q.w);
}

void print_energy(Energy e) {
  printf("Energy: total: %f, intra: %f, inter: %f, FE: %f \n", e.total, e.intra, e.inter, e.FE);
}

void print_coord(Coord c) {
  printf("Coord: x: %f, y: %f, z: %f \n", c.x, c.y, c.z);
}


void print_state(State s) {
  printf("State: \n");
  printf(" Translation: "); print_coord(s.T);
  printf(" Rotation: "); print_quat(s.Q);
  printf(" Torsions:  ");
  for(int i = 0; i < s.ntor; ++i) {
    if(i % 4 == 0)
      printf("\n    ");
    printf("%f ", s.tor[i]);
  }
  printf("\n");
  if(s.hasEnergy) {
    printf(" "); print_energy(s.e);
  }
  else {
    printf(" Energy undefined\n");
  }
  
  printf(" Center: "); print_coord(s.Center);
}


void print_molecule(Molecule* m) {
  printf("Num atoms: %d \n", m->natom);
  
  printf("Original coords: \n");
  for (int i=0; i<m->natom; ++i) {  
    printf("  %f %f %f \n", m->crdpdb[i][0], m->crdpdb[i][1], m->crdpdb[i][2]);
  }
  
  
  printf("Current coords: \n");
  for (int i=0; i<m->natom; ++i) {  
    printf("  %f %f %f \n", m->crd[i][0], m->crd[i][1], m->crd[i][2]);
  }

  
  printf("Atomstr: \n");
  for (int i=0; i<m->natom; ++i) {
    printf("  %s\n", m->atomstr[i]);
    }
  
  printf("Torsion vectors: \n");
  for (int i=0; i<m->S.ntor; ++i) {  
      printf("  %f %f %f \n", m->vt[i][0], m->vt[i][1], m->vt[i][2]);
  }
  
  printf("Torsion list: \n");
  for (int i=0; i<m->S.ntor; ++i) {
    printf("  ");
    for (int ii=0; ii < 25; ++ii) {
      printf("%02d ", m->tlist[i][ii]);
    }
  
    printf("\n");
  }
}
