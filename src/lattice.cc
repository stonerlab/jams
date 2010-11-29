#include "maths.h"
#include "consts.h"
#include "globals.h"
#include "lattice.h"
#include <libconfig.h++>
#include <map>
#include <cmath>
#include "array2d.h"
#include "array3d.h"
#include "array4d.h"
#include "sparsematrix.h"
#include <stdint.h>
#include <sstream>


enum SymmetryType {ISOTROPIC, UNIAXIAL, ANISOTROPIC, TENSOR, NOEXCHANGE};


void insert_interaction(int m, int n, int i,  Array2D<double> &jijval, SymmetryType exchsym) {
  using namespace globals;
// cusparse does not support symmetric matrix
#ifndef CUDA
  // only store lower triangle
  if( m > n ) {
#endif
    switch (exchsym) {
      case NOEXCHANGE:
        output.write("WARNING: Attempting to insert non existent exchange");
        break;
      case ISOTROPIC:
        Jij.insert(3*m+0,3*n+0,jijval(i,0)); // Jxx
        Jij.insert(3*m+1,3*n+1,jijval(i,0)); // Jyy
        Jij.insert(3*m+2,3*n+2,jijval(i,0)); // Jzz
        break;
      case UNIAXIAL:
        Jij.insert(3*m+0,3*n+0,jijval(i,0)); // Jxx
        Jij.insert(3*m+1,3*n+1,jijval(i,0)); // Jyy
        Jij.insert(3*m+2,3*n+2,jijval(i,1)); // Jzz
        break;
      case ANISOTROPIC:
        Jij.insert(3*m+0,3*n+0,jijval(i,0)); // Jxx
        Jij.insert(3*m+1,3*n+1,jijval(i,1)); // Jyy
        Jij.insert(3*m+2,3*n+2,jijval(i,2)); // Jzz
        break;
      case TENSOR:
        Jij.insert(3*m+0,3*n+0,jijval(i,0)); // Jxx
        Jij.insert(3*m+0,3*n+1,jijval(i,1)); // Jxy
        Jij.insert(3*m+0,3*n+2,jijval(i,2)); // Jxz
        
        Jij.insert(3*m+1,3*n+0,jijval(i,0)); // Jyx
        Jij.insert(3*m+1,3*n+1,jijval(i,1)); // Jyy
        Jij.insert(3*m+1,3*n+2,jijval(i,2)); // Jyz
        
        Jij.insert(3*m+2,3*n+0,jijval(i,0)); // Jzx
        Jij.insert(3*m+2,3*n+1,jijval(i,1)); // Jzy
        Jij.insert(3*m+2,3*n+2,jijval(i,2)); // Jzz
        break;
      default:
        jams_error("Undefined exchange symmetry. 1, 2, 3 or 9 components must be specified\n");
    }
#ifndef CUDA
  }
#endif
}


void Lattice::createFromConfig() {
  using namespace globals;

  Array4D<int> latt;
  Array3D<double> inter;
  Array2D<int> internbr;
  Array3D<double> jij;
  bool pbc[3] = {true,true,true};
  double r[3], p[3];
  int q[3];

  double unitcell[3][3];
  double unitcellinv[3][3];
  
  try {
    //-----------------------------------------------------------------
    //  Read basis from config
    //-----------------------------------------------------------------
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

    //-----------------------------------------------------------------
    //  Read atomic positions and types from config
    //-----------------------------------------------------------------
    const libconfig::Setting& atoms = config.lookup("lattice.unitcell.atoms");
    
    const int natoms = atoms.getLength();

    // map type_name -> int
    std::map<std::string,int>::iterator it_type;
                
    output.write("\nAtoms in unit cell\n");
    output.write("------------------\n");
    
    
    ntypes = 0;
    for (int n=0; n<natoms; ++n) {
      const std::string type_name=atoms[n][0];
      
      double pos[3];      
      for(int j=0; j<3; ++j) { 
        pos[j] = atoms[n][1][j]; 
      }

      output.write("%s %f %f %f\n",type_name.c_str(),pos[0],pos[1],pos[2]);

      it_type = atom_type_map.find(type_name);

      if (it_type == atom_type_map.end()) { 
        // type not found in map -> add to map
        atom_type_map.insert( std::pair<std::string,int>(type_name,ntypes) );
        ntypes++;
      }
    }

    output.write("\nUnique types found: %d\n",ntypes);

    type_count.resize(ntypes);

    for(int i=0; i<ntypes; ++i) {
      type_count[i] = 0;
    }

    //-----------------------------------------------------------------
    //  Read lattice properties from config
    //-----------------------------------------------------------------
    
    const libconfig::Setting& size = config.lookup("lattice.size");
    for(int i=0; i<3; ++i) {
      dim[i] = size[i];
    }
    
    output.write("Lattice size: %i %i %i\n",dim[0],dim[1],dim[2]);

    output.write("Lattice Periodic: ");

    const libconfig::Setting& periodic = config.lookup("lattice.periodic");
    for(int i=0; i<3; ++i) {
      pbc[i] = periodic[i];
      if(pbc[i]) {
        output.write("periodic ");
      } else {
        output.write("open ");
      }
    }
    output.write("\n");
    


    //-----------------------------------------------------------------
    //  Construct lattice 
    //-----------------------------------------------------------------

    const int maxatoms = dim[0]*dim[1]*dim[2]*natoms;

    latt.resize(dim[0],dim[1],dim[2],natoms);
    
    atom_type.reserve(maxatoms);

    int counter = 0;
    for (int x=0; x<dim[0]; ++x) {
      for (int y=0; y<dim[1]; ++y) {
        for (int z=0; z<dim[2]; ++z) {
          for (int n=0; n<natoms; ++n) {
            const std::string type_name = atoms[n][0];
            const int type_num = atom_type_map[type_name];
            atom_type.push_back(type_num);
            type_count[type_num]++;
            latt(x,y,z,n) = counter++;
          } // n
        } // z
      } // y
    } // x

    const unsigned int atomcount = counter;

    output.write("Total atoms: %i\n",atomcount);
    
    //-----------------------------------------------------------------
    //  Print lattice to file
    //-----------------------------------------------------------------

#ifdef DEBUG
    std::ofstream structfile;
    structfile.open("structure.xyz");
    structfile << nspins << "\n";
    structfile << "Generated by JAMS++\n";
    for (int x=0; x<dim[0]; ++x) {
      for (int y=0; y<dim[1]; ++y) {
        for (int z=0; z<dim[2]; ++z) {
          for (int n=0; n<natoms; ++n) {
            const std::string t = atoms[n][0];
            structfile << t <<"\t";
            q[0] = x; q[1] = y; q[2] = z;
            for(int i=0; i<3; ++i) {
              r[i] = 0.0;
              p[i] = atoms[n][1][i];
              for(int j=0; j<3; ++j) {
                r[i] += unitcell[j][i]*(q[j]+p[i]);
              }
            }
            structfile << 5*r[0] <<"\t"<< 5*r[1] <<"\t"<< 5*r[2] <<"\n";
          } // n
        } // z
      } // y
    } // x
    structfile.close();
#endif
    
    //-----------------------------------------------------------------
    //  Resize global arrays
    //-----------------------------------------------------------------
   
    nspins = atomcount;
    nspins3 = 3*nspins;

    s.resize(nspins,3);
    h.resize(nspins,3);
    w.resize(nspins,3);
    alpha.resize(nspins);
    mus.resize(nspins);
    gyro.resize(nspins);
    omega_corr.resize(nspins);

    //-----------------------------------------------------------------
    //  Initialise material parameters
    //-----------------------------------------------------------------
    const libconfig::Setting& mat = config.lookup("materials");

    for(int i=0; i<nspins; ++i) {
      int type_num = atom_type[i];
      double sinit[3];
      double norm=0.0;
      for(int j=0;j<3;++j) {
        sinit[j] = mat[type_num]["spin"][j]; 
        norm += sinit[j]*sinit[j];
      }
      norm = 1.0/sqrt(norm);
      for(int j=0;j<3;++j){
        
        s(i,j) = sinit[j]*norm;
        h(i,j) = 0.0;
        w(i,j) = 0.0;
      }

      // TODO: Move this to LLMS solver initialisation
      std::stringstream ss;
      ss << "materials.["<<type_num<<"].t_corr";

      if(config.lookupValue(ss.str(),omega_corr(i))){
        omega_corr(i) = 1.0/(gamma_electron_si*omega_corr(i));
      } else {
        omega_corr(i) = 0.0;
      }

      mus(i) = mat[type_num]["moment"];
      mus(i) = mus(i)*mu_bohr_si;
      
      alpha(i) = mat[type_num]["alpha"];

      gyro(i) = mat[type_num]["gyro"];
      gyro(i) = -gyro(i)/((1.0+alpha(i)*alpha(i)));
    }

    //-----------------------------------------------------------------
    //  Read exchange parameters from config
    //-----------------------------------------------------------------
    const libconfig::Setting& exch = config.lookup("exchange");
    
    int inter_total=0;

    bool jsym = config.lookup("lattice.jsym");

    const int inter_config = exch.getLength();

    // count number of interactions
    if(jsym == true) {
      for(int n=0;n<inter_config;++n) {
        double r[3];
        for(int j=0; j<3; ++j) {
          r[j] = exch[n][2][j];
        }
        std::sort(r,r+3);
        do {
          inter_total++;
        } while (next_point_symmetry(r));
      }
    } else {
      inter_total = inter_config;
    }
    output.write("Interactions in config: %d\n",inter_config);
    output.write("Total interactions (with symmetry): %d\n",inter_total);

    // Guess number of interactions to minimise vector reallocing
    // int the sparse matrix inserts
    int inter_guess = 3*atomcount*inter_total;
    
    // Find number of exchange tensor components specified in the
    // config
    int nexch;
    if(inter_total > 0) {
      nexch = exch[0][3].getLength();
    } else {
      nexch = 0;
    }

    // Select an exchange symmetry type based in the number of
    // components specified.
    SymmetryType exchsym=ISOTROPIC;
    switch (nexch) {
      case 0:
        exchsym = NOEXCHANGE;
        output.write("WARNING: No exchange found\n");
        break;
      case 1:
        exchsym = ISOTROPIC;
        output.write("Found isotropic exchange\n");
        break;
      case 2:
        exchsym = UNIAXIAL;
        output.write("Found uniaxial exchange\n");
        break;
      case 3:
        exchsym = ANISOTROPIC;
        output.write("Found anisotropic exchange\n");
        break;
      case 9:
        exchsym = TENSOR;
        output.write("Found tensorial exchange\n");
        inter_guess = nspins*inter_total*9;
        break;
      default:
        jams_error("Undefined exchange symmetry. 1, 2, 3 or 9 components must be specified\n");
    }
    
    // Resize interaction arrays
    inter.resize(natoms,inter_total,3);
    internbr.resize(natoms,inter_total);
    std::vector<int> nintype(natoms,0);

    // Resize global Jij matrix
    Jij.resize(3*atomcount,3*atomcount,inter_guess);

    //-----------------------------------------------------------------
    //  Read exchange tensor values from config
    //-----------------------------------------------------------------
    Array2D<double> jijval(inter_total,nexch);

    int inter_counter = 0;
    for(int n=0; n<inter_config; ++n) {
      double r[3];
      double p[3];
      double d_latt[3]={0.0,0.0,0.0};
      // read exchange tensor

      int atom_num_1 = exch[n][0];
      int atom_num_2 = exch[n][1];

      // count from zero
      atom_num_1--; atom_num_2--;

      for(int i=0; i<3; ++i) {
        // fractional vector within unit cell 
        p[i] = atoms[atom_num_2][1][i];
        // real space vector to neighbour
        r[i] = exch[n][2][i];
      }

      if(jsym==true) {
        std::sort(r,r+3);
        do {
          // place interaction vector in unitcell space
          for(int i=0; i<3; ++i) {
            d_latt[i] = 0.0;
            for(int j=0; j<3; ++j) {
              d_latt[i] += r[j]*unitcellinv[j][i];
            }
          }

          // store unitcell atom difference
          internbr(atom_num_1,nintype[atom_num_1]) = atom_num_2 - atom_num_1;

          // store interaction vectors
          for(int i=0;i<3; ++i){
            inter(atom_num_1,nintype[atom_num_1],i) = d_latt[i];
          }
    
          const std::string type_name=atoms[atom_num_1][0];

          int type_num = atom_type_map[type_name];

          // read tensor components
          for(int j=0; j<nexch; ++j) {
            double tmp = mat[type_num]["moment"];
            tmp *= mu_bohr_si;
            jijval(inter_counter,j) = exch[n][3][j];
            jijval(inter_counter,j) /= tmp;
          }

          nintype[atom_num_1]++;
          inter_counter++;
        } while (next_point_symmetry(r));
      } else {
        // place interaction vector in unitcell space
        for(int i=0; i<3; ++i) {
          d_latt[i] = 0.0;
          for(int j=0; j<3; ++j) {
            d_latt[i] += r[j]*unitcellinv[j][i];
          }
        }
        
        // store unitcell atom difference
        internbr(atom_num_1,nintype[atom_num_1]) = atom_num_2 - atom_num_1;
        
        // store interaction vectors
        for(int i=0;i<3; ++i){
          inter(atom_num_1,nintype[atom_num_1],i) = d_latt[i];
        }

        const std::string type_name=atoms[atom_num_1][0];

        int type_num = atom_type_map[type_name];

        // read tensor components
        for(int j=0; j<nexch; ++j) {
          double tmp = mat[type_num]["moment"];
          tmp *= mu_bohr_si;
          jijval(inter_counter,j) = exch[n][3][j];
          jijval(inter_counter,j) /= tmp;
        }
        inter_counter++;
        nintype[atom_num_1]++;
      }

    }

    //-----------------------------------------------------------------
    //  Create interaction matrix
    //-----------------------------------------------------------------

      double encut = 1e-25;  // energy cutoff
      double p[3], pnbr[3];
      int v[3], q[3], qnbr[3];

      counter = 0;
      for (int x=0; x<dim[0]; ++x) {
        for (int y=0; y<dim[1]; ++y) {
          for (int z=0; z<dim[2]; ++z) {
            for (int n=0; n<natoms; ++n) {

              const int atom = latt(x,y,z,n);
              const int type_num = atom_type[atom];

              assert(atom < nspins);

              q[0] = x; q[1] = y; q[2] = z;
              
              for(int j=0; j<3; ++j) {
                p[j] = atoms[n][1][j];
              }
            
              double tmp = mat[type_num]["moment"];
              tmp *= mu_bohr_si;

              // anisotropy value
              double anival = mat[type_num]["anisotropy"][1];
		anival /= tmp;

              for(int i=0;i<3;++i) {
              // easy axis
                double ei = mat[type_num]["anisotropy"][0][i];
              // magnitude
                double di = 2.0*anival*ei ; 
              // insert if above encut
                if(fabs(di) > encut ){
                 Jij.insert(3*atom+i,3*atom+i, di );
               }
              }

              for(int i=0; i<nintype[n]; ++i) {
                int m = (internbr(n,i)+n)%natoms;
                
                for(int j=0; j<3; ++j) {
                  pnbr[j] = atoms[m][1][j];
                  qnbr[j] = floor(inter(n,i,j)-pnbr[j]+0.5);
                }

                for(int j=0; j<3; ++j) {
                  v[j] = q[j]+qnbr[j];
                  if(pbc[j] == true) {
                    v[j] = (dim[j]+v[j])%dim[j];
                  }
                }
                bool idxcheck = true;
                for(int j=0;j<3;++j) {
                  if(v[j] < 0 || !(v[j] < dim[j])) {
                    idxcheck = false;
                  }
                }

                if(idxcheck == true) {
                  int nbr = latt(v[0],v[1],v[2],m);
                  assert(nbr < nspins);
                  assert(atom != nbr);
                  insert_interaction(atom,nbr,i,jijval,exchsym);
                  if( atom > nbr ) {
                    counter++;
                  }
                }
              }
            } // n
          } // z
        } // y
      } // x


    output.write("\nInteraction count: %i\n", counter);
    output.write("Jij memory (COO): %f MB\n",Jij.memorySize());
    output.write("Converting COO to CSR INPLACE\n");
    Jij.coocsrInplace();
    output.write("Jij memory (CSR): %f MB\n",Jij.memorySize());


  
  } // try
  catch(const libconfig::SettingNotFoundException &nfex) {
    jams_error("Setting '%s' not found",nfex.getPath());
  }

}
