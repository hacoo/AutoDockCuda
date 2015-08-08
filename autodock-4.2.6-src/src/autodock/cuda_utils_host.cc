
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
#include <cstring>

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


int getNumAtoms(Molecule* m){
  return m->natom;
}

char* getAtomString(Molecule* m, int n) {
  char* thisString = new char[strlen(m->atomstr[n])+1];
  strcpy(thisString, m->atomstr[n]);
  return thisString;
}

char** getAtomStringArray(Molecule* m) {
  // Returns a 'ragged array' of atom strings 
  // for the molecule m. The number of strings in the array 
  // is the number of atoms in the molecule, and each string 
  // has a max length of MAX_CHARS
  
  int numatoms = m->natom;
  char** thisArray = new char*[numatoms];
  for(int i=0; i<numatoms; ++i){
    thisArray[i] = getAtomString(m, i);
  }
  return thisArray;
}


void freeAtomStringArray(char** a, int numatoms) {
  // Free the atom string array a contains numatoms
  // lines.
  if(!a)
    return;
  for (int i=0; i<numatoms; ++i) {
    if(a[i])
      delete a[i];
  }
}


double* getTorsions(Molecule* m, int ntors) {
  // Returns array of torsion vectors from molecule m.
  // Each has SPACE components (probably 3)
  double* torsions = new double[ntors*SPACE];
  memcpy(torsions, m->vt, sizeof(double)*ntors*SPACE);
  return torsions;
}


double* getAtomCrds(Molecule* m) {
  // Returns array of coordinate vectors for each atom in m.
  // Each vector has SPACE components (probably 3)
  int natoms = m->natom;
  double* crds = new double[natoms*SPACE];
  memcpy(crds, m->crdpdb, sizeof(double)*natoms*SPACE);
  return crds;
}


int* getTorsionRootList(Molecule* m, int ntors) {
  // Returns an array of torsion root lists.
  // Each torsion has a list of root atoms that must be 
  // evaluated before this torsion is evaulated. The torsion root
  // list reprents this.

  int natoms = m->natom;
  int* rootlist = new int[natoms*ntors];
  for(int i=0; i<ntors; ++i) {
    for (int ii=0; ii<natoms; ++ii) {
	rootlist[i*natoms+ii] = m->tlist[i][ii];
      }
  }
  return rootlist;
}
