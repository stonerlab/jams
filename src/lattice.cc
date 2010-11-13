#include "maths.h"
#include "consts.h"
#include "globals.h"
#include "lattice.h"
#include <libconfig.h++>
#include <map>
#include "array2d.h"
#include "array3d.h"
#include "array4d.h"
#include "sparsematrix.h"
#include <stdint.h>
#include <sstream>

//#include <metis.h>

enum SymmetryType {ISOTROPIC, UNIAXIAL, ANISOTROPIC, TENSOR, NOEXCHANGE};

#ifdef MPI
extern "C" { 
#include <metis.h>
} 
#endif

void insert_interaction(int m, int n, int i,  Array2D<double> &jijval, SymmetryType exchsym) {
  using namespace globals;
  // only store lower triangle
  if( m > n ) {
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
  }
}


void Lattice::createFromConfig() {
  using namespace globals;

  /*
  SparseMatrix<double> Test(5,5,13);

  Test.insert(0,0,1);
  Test.insert(0,1,-1);
  Test.insert(0,3,-3);
  //Test.insert(1,0,-2);
  Test.insert(1,1,5);
  Test.insert(2,2,4);
  Test.insert(2,3,6);
  Test.insert(3,3,7);
  Test.insert(2,4,4);
  //Test.insert(3,0,-4);
  Test.insert(3,2,2);
  //Test.insert(4,1,8);
  Test.insert(4,4,-5);

  for(int i=0;i<13;++i) {
    double *val = Test.ptrVal();
    output.write("%f ",val[i]);
  }
  output.write("\n");
  
  for(int i=0;i<13;++i) {
    int *val = Test.ptrCol();
    output.write("%d ",val[i]);
  }
  output.write("\n");
  
  for(int i=0;i<13;++i) {
    int *val = Test.ptrRow();
    output.write("%d ",val[i]);
  }
  output.write("\n");

  output.write("Convert to CSR\n");

  Test.coocsrInplace();
  for(int i=0;i<13;++i) {
    double *val = Test.ptrVal();
    output.write("%f ",val[i]);
  }
  output.write("\n");
  
  for(int i=0;i<13;++i) {
    int *val = Test.ptrCol();
    output.write("%d ",val[i]);
  }
  output.write("\n");
  
  for(int i=0;i<5;++i) {
    int *val = Test.ptrB();
    output.write("%d ",val[i]);
  }
  output.write("\n");
  
  for(int i=0;i<5;++i) {
    int *val = Test.ptrE();
    output.write("%d ",val[i]);
  }
  output.write("\n");
  output.write("\n");

  char descra[6]={'S','U','C','D','E'};
  char trans[1] = {'N'};
  double xvec[5] = {1,2,3,4,5};
  double yvec[5] = {0,0,0,0,0};
  for(int i=0;i<5;++i) {
    output.write("%f ",yvec[i]);
  }
  output.write("\n");
  jams_dcsrmv(trans,5,5,1.0,descra,Test.ptrVal(),Test.ptrCol(),
      Test.ptrB(), Test.ptrE(),xvec,1.0,yvec);
  
  for(int i=0;i<5;++i) {
    output.write("%f ",yvec[i]);
  }
  output.write("\n");



  exit(0);
*/

  Array4D<int> latt;
  Array3D<int> inter;
  Array3D<double> jij;
  bool pbc[3] = {true,true,true};

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

    output.write("Total atoms: %i\n",atomcount);
    nspins = atomcount;
    nspins3 = 3*nspins;
   
    ///////////////////////// Read Exchange /////////////////////////
    const libconfig::Setting& exch = config.lookup("exchange");
    const int intertot = exch.getLength();

    const libconfig::Setting& mat = config.lookup("materials");


    s.resize(nspins,3);
    h.resize(nspins,3);
    w.resize(nspins,3);
    
    alpha.resize(nspins);
    mus.resize(nspins);
    gyro.resize(nspins);
    omega_corr.resize(nspins);

    // init spins
    for(int i=0; i<nspins; ++i) {
      int t1 = atom_type[i];
      double sinit[3];
      double norm=0.0;
      for(int j=0;j<3;++j) {
        sinit[j] = mat[t1]["spin"][j]; 
        norm += sinit[j]*sinit[j];
      }
      norm = 1.0/sqrt(norm);
      for(int j=0;j<3;++j){
        
        s(i,j) = sinit[j]*norm;
        h(i,j) = 0.0;
        w(i,j) = 0.0;
      }

      if(config.lookupValue("materials.t_corr",omega_corr(i))){
        omega_corr(i) = 1.0/(gamma_electron_si*omega_corr(i));
      } else {
        omega_corr(i) = 0.0;
      }

      mus(i) = mat[t1]["moment"];
      mus(i) = mus(i)*mu_bohr_si;
      alpha(i) = mat[t1]["alpha"];
      gyro(i) = mat[t1]["gyro"];
      gyro(i) = -gyro(i)/((1.0+alpha(i)*alpha(i))*(mus(i)));
    }
     
    int nexch;
    if(intertot > 0) {
      nexch = exch[0][3].getLength();
    } else {
      nexch = 0;
    }


      int inter_guess = atomcount*intertot*3;


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
          inter_guess = atomcount*intertot*9;
          break;
        default:
          jams_error("Undefined exchange symmetry. 1, 2, 3 or 9 components must be specified\n");
      }
      
      inter.resize(natoms,intertot,4);
      std::vector<int> nintype(natoms,0);

      Jij.resize(3*atomcount,3*atomcount,inter_guess);

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
          //if( fabs(r[i]-p[i]-v[i]) > 0.01) {
          //  jams_error("Exchange lattice mismatch on interaction: %i (r[%i]-p[%i] = %f, v[%i] = %i)",n+1,i,i,r[i]-p[i],i,v[i]);
          //}
        }
        for(int j=0; j<nexch; ++j) {
          jijval(n,j) = exch[n][3][j];
          jijval(n,j) *= 0.5*jijval(n,j);
        }
        for(int i=0;i<4; ++i){
          inter(t1,nintype[t1],i) = v[i];
        }
        nintype[t1]++;
      }
      
      /////////////////// Create interaction list /////////////////////////

      // i,j neighbour list for system partitioning
  //    SparseMatrix<int> nbr_list(atomcount,atomcount,inter_guess);

      double encut = 1e-25;  // energy cutoff
      bool jsym = config.lookup("lattice.jsym");
      counter = 0;
      for (int x=0; x<dim[0]; ++x) {
        for (int y=0; y<dim[1]; ++y) {
          for (int z=0; z<dim[2]; ++z) {
            for (int n=0; n<natoms; ++n) {
              
              const int atom = latt(x,y,z,n);
              const int t1 = atom_type[atom];


              // insert anisotropy
              double anival = mat[t1]["anisotropy"][1];

              for(int i=0;i<3;++i) {
                double ei = mat[t1]["anisotropy"][0][i];
                double di = anival*ei ; 
                if(fabs(di) > encut ){
                  Jij.insert(3*atom+i,3*atom+i,di);
                }
              }



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
                      v[j] = r[j]+q[j];
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
                      insert_interaction(atom,nbr,i,jijval,exchsym);
  //                  nbr_list.insert(atom,nbr,1);
                      counter++;
                    }
                  } while (next_point_symmetry(q));
                }
                else {
                  for(int j=0; j<3; ++j) {
                    v[j] = r[j]+q[j];
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
                   insert_interaction(atom,nbr,i,jijval,exchsym);
                   insert_interaction(nbr,atom,i,jijval,exchsym);
    //               nbr_list.insert(atom,nbr,1);
                   counter++;
                  }
                }
              }
            } // n
          } // z
        } // y
      } // x

//    nbr_list.coocsr();

#ifdef MPI
    output.write("Partitioning the interaction graph\n");
    int options[5] = {0,3,1,1,0};
    int volume = 0;
    int wgtflag = 0;
    int numflag = 0;
    int nparts = 4;
    int nvertices = static_cast<int>(atomcount); //nbr_list.nonzero();
    std::vector<int> part(nvertices,0);

    //nbr_list.printCSR();
    output.write("Parts: %i\n",nparts);

    output.write("Calling METIS\n");
    METIS_PartGraphVKway(&nvertices, nbr_list.ptrRow(), nbr_list.ptrCol(), NULL, NULL, 
        &wgtflag, &numflag, &nparts, options, &volume, &part[0]);

    output.write("Communication volume: %i\n",volume);
    
    for (int p=0; p<nparts; ++p) {
      std::stringstream pname;
      pname << "part" << p << ".out";
      std::string filename = pname.str();
      std::ofstream pfile(filename.c_str());
      for (int x=0; x<dim[0]; ++x) {
        for (int y=0; y<dim[1]; ++y) {
          for (int z=0; z<dim[2]; ++z) {
            for (int n=0; n<natoms; ++n) {
              if(part[latt(x,y,z,n)] == p){
                pfile << x <<"\t"<< y <<"\t"<< z <<"\t"<< n <<"\t"<<part[latt(x,y,z,n)]<<"\n";
              }
            }
          }
        }
      }
    }



    //for(int i=0;i<nvertices;++i) {
    //  output.write("%i\n",part[i]);
    //}
#endif

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
