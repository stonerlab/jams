#include "maths.h"
#include "globals.h"
#include "lattice.h"
#include <libconfig.h++>
#include <map>
#include <array3d.h>
#include <array4d.h>

enum SymmetryType {ISOTROPIC, ANISOTROPIC, TENSOR};

// interactions should be inter[type][ninter][loc] = jij where loc = {dx,dy,dz,natom}
// atoms should be atoms[x][y][z][natom] = atomnumber
// jij should be jij[type][ninter][ii] = jijval where ii is 0-8 (9
// tensor components

void Lattice::createFromConfig() {

  Array4D<int> latt;
  Array3D<int> inter;
  Array3D<double> jij;

  double unitcell[3][3];
  double unitcellinv[3][3];
  
  try {
    ///////////////////////// read basis /////////////////////////
    const libconfig::Setting& basis = config.lookup("lattice.unitcell.basis");

    for(int i=0; i<3; ++i) {
      for(int j=0; j<3; ++j) {
        unitcell[i][j] = basis[i][j];
      }
    }
  
    matrix_invert(unitcell,unitcellinv);

    output.write("\nLattice translation vectors\n");
    output.write("---------------------------\n");
    for(int i=0; i<3; ++i) {
      output.write("%f %f %f\n",unitcell[i][0],unitcell[i][1],unitcell[i][2]);
    }
    
    output.write("\nInverse lattice vectors\n");
    output.write("---------------------------\n");
    for(int i=0; i<3; ++i) {
      output.write("%f %f %f\n",unitcellinv[i][0],unitcellinv[i][1],unitcellinv[i][2]);
    }

    ///////////////////////// read atoms /////////////////////////
    const libconfig::Setting& atoms = config.lookup("lattice.unitcell.atoms");
    int natoms = atoms.getLength();              
                
    output.write("\nAtoms in unit cell\n");
    output.write("------------------\n");
    
    std::map<std::string,int>::iterator tit;
    
    ntypes = 0;
    for (int n=0; n<natoms; ++n) {
      std::string t=atoms[n][0];
      
      double pos[3];      
      for(int i=0; i<3; ++i) { 
        pos[i] = atoms[n][1][i]; 
      }
      output.write("%s %f %f %f\n",t.c_str(),pos[0],pos[1],pos[2]);

      tit = atom_type_map.find(t);
      if (tit == atom_type_map.end()) {
        atom_type_map.insert( std::pair<std::string,int>(t,ntypes) );
        ntypes++;
      }
    }
    output.write("\nUnique types found: %d\n",ntypes);

    
    const libconfig::Setting& size = config.lookup("lattice.size");
    for(int i=0; i<3; ++i) {
      dim[i] = size[i];
    }
    output.write("Lattice size: %i %i %i\n",dim[0],dim[1],dim[2]);

    /////////////////// construct lattice //////////////////////
    // number of atoms will be dim[0]*dim[1]*dim[2]*natomsincell


    const int maxatoms = dim[0]*dim[1]*dim[2]*natoms;

    latt.resize(dim[0],dim[1],dim[2],natoms);
    
    atom_type.reserve(maxatoms);

    int counter = 0;
    for (int x=0; x<dim[0]; ++x) {
      for (int y=0; y<dim[1]; ++y) {
        for (int z=0; z<dim[2]; ++z) {
          for (int n=0; n<natoms; ++n) {
            const std::string t = atoms[n][0];
            atom_type.push_back(atom_type_map[t]);
            latt(x,y,z,n) = counter++;
          } // n
        } // z
      } // y
    } // x

    output.print("Total atoms: %i\n",counter);

    ///////////////////////// Read Exchange /////////////////////////
    const libconfig::Setting& exch = config.lookup("exchange");
    const int intertot = exch.getLength();
    const int nexch = exch[0][3].getLength();

    SymmetryType exchsym;
    if(nexch == 1) {
      exchsym = ISOTROPIC;
    } 
    else if (nexch == 2) {
      exchsym = ANISOTROPIC;
    } 
    else if (nexch == 9) {
      exchsym = TENSOR;
    } 
    else {
      jams_error("Unknown exchange symmetry, 1, 2 or 9 components must be specified");
    }

    inter.resize(natoms,intertot,4);
    std::vector<int> nintype(natoms,0);


    double jijval[nexch];
    double r[3];
    double p[3];
    int v[4];
    for(int n=0; n<intertot; ++n) {
      // read exchange tensor
      int t1 = exch[n][0];
      int t2 = exch[n][1];
      t1--; t2--;
      v[3] = t2-t1;
      for(int i=0; i<3; ++i) {
        // p is the vector of the exchange partner within the
        // unit cell (real space)
        p[i] = atoms[t2][1][i];
        // r is the vector to the unitcell containing the exchange
        // partner (real space)
        r[i] = exch[n][2][i];
        // relative interger coords to unit cell containing interaction
        v[i] = floor((r[i]-p[i])+0.5);

        // check exchange-lattice alignment
#ifndef NOCHECKING
        if( fabs(r[i]-p[i]-v[i]) > 0.01) {
          jams_error("Exchange lattice mismatch on interaction: %i",n+1);
        }
#endif
      }
      for(int j=0; j<nexch; ++j) {
        jijval[j] = exch[n][3][j];
      }
      for(int i=0;i<4; ++i){
        inter(t1,nintype[t1],i) = v[i];
      }
      nintype[t1]++;
    }
    
    /////////////////// Create interaction list /////////////////////////
    counter = 0;
    for (int x=0; x<dim[0]; ++x) {
      for (int y=0; y<dim[1]; ++y) {
        for (int z=0; z<dim[2]; ++z) {
          for (int n=0; n<natoms; ++n) {
            const int atom = latt(x,y,z,n);
            const int t1 = atom_type[atom];
            const int r[3] = {x,y,z};  // current lattice point
            int v[3];
            int q[3];
//            output.write("\n%i: ",atom);
            for(int i=0; i<nintype[t1]; ++i) {
              for(int j=0; j<3; ++j) {
                q[j] = inter(t1,i,j);
              }
              int m = inter(t1,i,3);

              // loop symmetry points
              std::sort(q,q+3);
              do {
                for(int j=0; j<3; ++j) {
                  v[j] = (dim[j]+r[j]+q[j])%dim[j];
                }
                //output.write("%i %i\n",atom, latt(v[0],v[1],v[2],n+m));
                counter++;
              } while (next_point_symmetry(q));
            }
          } // n
        } // z
      } // y
    } // x
                
    output.write("\nInteraction count: %i\n", counter);

  
  } // try
  catch(const libconfig::SettingNotFoundException &nfex) {
    jams_error("Setting '%s' not found",nfex.getPath());
  }

}
