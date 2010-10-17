#include "maths.h"
#include "globals.h"
#include "lattice.h"
#include <libconfig.h++>
#include <map>
#include "array2d.h"
#include "array3d.h"
#include "array4d.h"

enum SymmetryType {ISOTROPIC, UNIAXIAL, ANISOTROPIC, TENSOR};

// interactions should be inter[type][ninter][loc] = jij where loc = {dx,dy,dz,natom}
// atoms should be atoms[x][y][z][natom] = atomnumber
// jij should be jij[type][ninter][ii] = jijval where ii is 0-8 (9
// tensor components

void insert_interaction(int i, int j, int n, Array2D<double> &val, SymmetryType sym) {
  using namespace globals;
  /*
  switch (sym) {
    case ISOTROPIC:
      jijxx->insert(i,j,val(n,0));
      break;
    case UNIAXIAL:
      jijxx->insert(i,j,val(n,0));
      jijzz->insert(i,j,val(n,1));
      break;
    case ANISOTROPIC:
      jijxx->insert(i,j,val(n,0));
      jijyy->insert(i,j,val(n,1));
      jijzz->insert(i,j,val(n,2));
      break;
    case TENSOR:
      jijxx->insert(i,j,val(n,0));
      jijxy->insert(i,j,val(n,1));
      jijxz->insert(i,j,val(n,2));
      jijyx->insert(i,j,val(n,3));
      jijyy->insert(i,j,val(n,4));
      jijyz->insert(i,j,val(n,5));
      jijzx->insert(i,j,val(n,6));
      jijzy->insert(i,j,val(n,7));
      jijzz->insert(i,j,val(n,8));
      break;
    default:
      jams_error("Undefined symmetry when->inserting interaction");
  }
  */
}

void Lattice::createFromConfig() {
  using namespace globals;

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

    const unsigned int atomcount = counter;

    output.print("Total atoms: %i\n",atomcount);

    ///////////////////////// Read Exchange /////////////////////////
    const libconfig::Setting& exch = config.lookup("exchange");
    const int intertot = exch.getLength();
    const int nexch = exch[0][3].getLength();

    const int inter_guess = atomcount*intertot;
    SymmetryType exchsym=ISOTROPIC;
    /*
    switch (nexch) {
      case 1:
        exchsym = ISOTROPIC;
        output.write("Found isotropic exchange\n");
        jijxx = new SparseMatrix(atomcount,atomcount,inter_guess);
        jijyy = jijxx;
        jijzz = jijxx;
        break;
      case 2:
        exchsym = UNIAXIAL;
        output.write("Found uniaxial exchange\n");
        jijxx = new SparseMatrix(atomcount,atomcount,inter_guess);
        jijyy = jijxx;
        jijzz = new SparseMatrix(atomcount,atomcount,inter_guess);
        break;
      case 3:
        exchsym = ANISOTROPIC;
        output.write("Found anisotropic exchange\n");
        jijxx = new SparseMatrix(atomcount,atomcount,inter_guess);
        jijyy = new SparseMatrix(atomcount,atomcount,inter_guess);
        jijzz = new SparseMatrix(atomcount,atomcount,inter_guess);
        break;
      case 9:
        exchsym = TENSOR;
        jijxx = new SparseMatrix(atomcount,atomcount,inter_guess);
        jijyy = new SparseMatrix(atomcount,atomcount,inter_guess);
        jijzz = new SparseMatrix(atomcount,atomcount,inter_guess);
        output.write("Found tensorial exchange\n");
        break;
      default:
        jams_error("Undefined exchange symmetry. 1, 2, 3 or 9 components must be specified\n");
    }
    jijxy = new SparseMatrix(atomcount,atomcount,inter_guess);
    jijxz = new SparseMatrix(atomcount,atomcount,inter_guess);

    jijyx = new SparseMatrix(atomcount,atomcount,inter_guess);
    jijyz = new SparseMatrix(atomcount,atomcount,inter_guess);

    jijzx = new SparseMatrix(atomcount,atomcount,inter_guess);
    jijzy = new SparseMatrix(atomcount,atomcount,inter_guess);
*/
    inter.resize(natoms,intertot,4);
    std::vector<int> nintype(natoms,0);


    Array2D<double> jijval(intertot,nexch);
    for(int n=0; n<intertot; ++n) {
    double r[3];
    double p[3];
    int v[4];
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
          jams_error("Exchange lattice mismatch on interaction: %i (r[%i]-p[%i] = %f, v[%i] = %i)",n+1,i,i,r[i]-p[i],i,v[i]);
        }
#endif
      }
      for(int j=0; j<nexch; ++j) {
        jijval(n,j) = exch[n][3][j];
      }
      for(int i=0;i<4; ++i){
        inter(t1,nintype[t1],i) = v[i];
      }
      nintype[t1]++;
    }
    
    /////////////////// Create interaction list /////////////////////////
    bool jsym = config.lookup("lattice.jsym");
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
              if(jsym==true) {
                std::sort(q,q+3);
                do {
                  for(int j=0; j<3; ++j) {
                    v[j] = (dim[j]+r[j]+q[j])%dim[j];
                  }
                //output.write("%i %i\n",atom, latt(v[0],v[1],v[2],n+m));
                  insert_interaction(atom,m,i,jijval,exchsym);
                  counter++;
                } while (next_point_symmetry(q));
              }
              else {
               for(int j=0; j<3; ++j) {
                 v[j] = (dim[j]+r[j]+q[j])%dim[j];
               }
                //output.write("%i %i\n",atom, latt(v[0],v[1],v[2],m));
               insert_interaction(atom,m,i,jijval,exchsym);
               counter++;
              }
            }
          } // n
        } // z
      } // y
    } // x
                
    output.write("\nInteraction count: %i\n", counter);
    
    /*
    switch (exchsym) {
      case ISOTROPIC:
        jijxx->coocsr();
        break;
      case UNIAXIAL:
        jijxx->coocsr();
        jijzz->coocsr();
        break;
      case ANISOTROPIC:
        jijxx->coocsr();
        jijyy->coocsr();
        jijzz->coocsr();
        break;
      case TENSOR:
        jijxx->coocsr();
        jijyy->coocsr();
        jijzz->coocsr();
        
        break;
      default:
        jams_error("Undefined exchange symmetry. 1, 2, 3 or 9 components must be specified\n");
    }
    
    jijxy->coocsr();
    jijxz->coocsr();
    jijyx->coocsr();
    jijyz->coocsr();
    jijzx->coocsr();
    jijzy->coocsr();
*/
  
  } // try
  catch(const libconfig::SettingNotFoundException &nfex) {
    jams_error("Setting '%s' not found",nfex.getPath());
  }

}
