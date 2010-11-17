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

    type_count.resize(ntypes);
    for(int i=0; i<ntypes; ++i) {
      type_count[i] = 0;
    }

    
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
            const int t1 = atom_type_map[t];
            atom_type.push_back(t1);
            type_count[t1]++;
            latt(x,y,z,n) = counter++;
          } // n
        } // z
      } // y
    } // x

    const unsigned int atomcount = counter;

    output.write("Total atoms: %i\n",atomcount);
    nspins = atomcount;
    nspins3 = 3*nspins;
    
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
   
    ///////////////////////// Read Exchange /////////////////////////
    const libconfig::Setting& exch = config.lookup("exchange");
    const int intertot = exch.getLength();
    output.write("Interactions in config: %d\n",intertot);

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

      std::stringstream ss;
      ss << "materials.["<<t1<<"].t_corr";

      if(config.lookupValue(ss.str(),omega_corr(i))){
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
      
      inter.resize(ntypes,intertot,3);
      internbr.resize(ntypes,intertot);
      std::vector<int> nintype(ntypes,0);

      Jij.resize(3*atomcount,3*atomcount,inter_guess);

      Array2D<double> jijval(intertot,nexch);


      for(int n=0; n<intertot; ++n) {
      double r[3];
      double p[3];
      double d_latt[3]={0.0,0.0,0.0};
      //int v[4];
        // read exchange tensor
        int p1 = exch[n][0];
        int p2 = exch[n][1];
        p1--; p2--;
        //v[3] = p2-p1;

        for(int i=0; i<3; ++i) {
          // p is the vector of the exchange partner within the
          // unit cell (real space)
          p[i] = atoms[p2][1][i];
          // r is the vector to the unitcell containing the exchange
          // partner (real space)
          r[i] = exch[n][2][i];
        }
        
        // lattice integer offsets
        for(int i=0; i<3; ++i) {
          d_latt[i] = 0.0;
          for(int j=0; j<3; ++j) {
            d_latt[i] += r[j]*unitcellinv[j][i];
          }
        }

        //for(int i=0; i<3; ++i) {
        //  v[i] = floor(d_latt[i]+0.5);
        //}

        for(int j=0; j<nexch; ++j) {
          jijval(n,j) = exch[n][3][j];
        }
        
        std::string tname = atoms[p1][0];
        int t1 = atom_type_map[tname];
        internbr(t1,nintype[t1]) = p2-p1;
        for(int i=0;i<3; ++i){
          inter(t1,nintype[t1],i) = d_latt[i];
        }
//        output.print("t1:%d n:%d dx:%f dy:%f dz:%f vx:%d vy:%d vz:%d k:%d \n",t1,nintype[t1],d_latt[0],d_latt[1],d_latt[2]);

//        std::cerr<<v[0]<<"\t"<<v[1]<<"\t"<<v[2]<<"\t"<<v[3]<<"\n";
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
              assert(atom < nspins);

              const std::string tname = atoms[n][0];
              
              double p[3]={0.0,0.0,0.0}; // vector within unit cell
              double r[3]={0.0,0.0,0.0}; // realspace vector
              int q[3] = {x,y,z}; // integer lattice
              
                for(int j=0; j<3; ++j) {
                  p[j] = atoms[n][1][j];
                }


              //for(int i=0; i<3; ++i) {
              //  for(int j=0; j<3; ++j) {
              //    r[i] += unitcell[j][i]*(q[j]+p[j]);
              //  }
              //}

              // anisotropy value
              double anival = mat[t1]["anisotropy"][1];

              for(int i=0;i<3;++i) {
                // easy axis
                double ei = mat[t1]["anisotropy"][0][i];
                // magnitude
                double di = 2.0*anival*ei ; 
                // insert if above encut
                if(fabs(di) > encut ){
                  Jij.insert(3*atom+i,3*atom+i,di);
                }
              }



              //const int r[3] = {x,y,z};  // coord of current unit cell
              //int q[3]; // relative coordiante
              int v[3]; // total coordinate
              double pnbr[3];
              double qnbr[3];
              for(int i=0; i<nintype[t1]; ++i) {
                int m = (internbr(t1,i)+n)%natoms;
                
                for(int j=0; j<3; ++j) {
                  pnbr[j] = atoms[m][1][j];
                  qnbr[j] = floor(inter(t1,i,j)-pnbr[j]+0.5);
                }


                // loop symmetry points
                if(jsym==true) {
                  //output.write("WARNING: jsym is currently buggy\n");
                  std::sort(q,q+3);
                  do {
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
                      insert_interaction(atom,nbr,i,jijval,exchsym);
  //                  nbr_list.insert(atom,nbr,1);
                      if( atom > nbr ) {
//                        std::cout<<atom<<"\t"<<nbr<<"\n";
                        counter++;
                      }
//                      } else {
//                        std::cerr<<atom<<"\t"<<nbr<<"\n";
//                      }
                    }
                  } while (next_point_symmetry(q));
                }
                else {
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
    //               nbr_list.insert(atom,nbr,1);
                    if( atom > nbr ) {
                      //std::cout<<atom<<"\t"<<nbr<<"\n";
                      counter++;
                    } //else {
                      //std::cerr<<atom<<"\t"<<nbr<<"\n";
                   // }
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
