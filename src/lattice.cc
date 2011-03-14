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
#include <iostream>
#include <fstream>
#include <string>


///
/// @brief  Read basis vectors from config file.
///
void readBasis (const libconfig::Setting &cfgBasis, double unitcell[3][3], double unitcellInv[3][3]) 
{
  //  Read basis from config
  using namespace globals;

  for(int i=0; i<3; ++i) {
    for(int j=0; j<3; ++j) {
      unitcell[i][j] = cfgBasis[i][j];
    }
  }

  matrix_invert(unitcell,unitcellInv);

  output.write("\nLattice translation vectors\n---------------------------\n");
  for(int i=0; i<3; ++i) { 
    output.write("%f %f %f\n",unitcell[i][0],unitcell[i][1],unitcell[i][2]); 
  }
  
  output.write("\nInverse lattice vectors\n---------------------------\n");
  for(int i=0; i<3; ++i) { 
    output.write("%f %f %f\n",unitcellInv[i][0],unitcellInv[i][1],unitcellInv[i][2]); 
  }
}

///
/// @brief  Read atom positions and types from config file.
///
void readAtoms(Array<int> &unitCellTypes, Array2D<double> &unitCellPositions, int &nAtoms, int &nTypes, std::map<std::string,int> &atomTypeMap) {
  //  Read atomic positions and types from config
  

  std::ifstream positionFile("pos.in");

  nAtoms = 0;
  for (std::string line; getline(positionFile,line);) {
    nAtoms++;
  }

  unitCellTypes.resize(nAtoms);
  unitCellPositions.resize(nAtoms,3);

  output.write("\nAtoms in unit cell\n------------------\n");
  
  std::map<std::string,int>::iterator it_type;
  nTypes = 0;
  
  std::string typeName;
  
  positionFile.clear();
  positionFile.seekg(0,std::ios::beg);

  for(int n=0; n<nAtoms; ++n) {

    std::string line;
    
    getline(positionFile,line);

    std::stringstream is(line);
    
    is >> typeName;
    for(int j=0; j<3; ++j) { 
      is >> unitCellPositions(n,j);
    }

    output.write("%s %f %f %f\n",typeName.c_str(), unitCellPositions(n,0), unitCellPositions(n,1), unitCellPositions(n,2));

    it_type = atomTypeMap.find(typeName);
    if (it_type == atomTypeMap.end()) { 
      // type not found in map -> add to map
      // map type_name -> int
      
      unitCellTypes(n) = nTypes;

      atomTypeMap.insert( std::pair<std::string,int>(typeName,nTypes) );
      nTypes++;
    } else {
      unitCellTypes(n) = atomTypeMap[typeName];
    }
  }
  output.write("\nUnique types found: %d\n",nTypes);

  positionFile.close();
}

///
/// @brief  Read lattice parameters from config file.
///
void readLattice(const libconfig::Setting &cfgLattice, std::vector<int> &dim, bool pbc[3]) {
  //  Read lattice properties from config
  
  for(int i=0; i<3; ++i) { 
    dim[i] = cfgLattice["size"][i]; 
  }
  output.write("Lattice size: %i %i %i\n",dim[0],dim[1],dim[2]);

  output.write("Lattice Periodic: ");

  for(int i=0; i<3; ++i) {
    pbc[i] = cfgLattice["periodic"][i];
    if (pbc[i]) { 
      output.write("periodic "); 
    }
    else { 
      output.write("open "); 
    }
  }
  output.write("\n");
}

///
/// @brief  Create lattice on numbered spin locations.
///
void createLattice(const libconfig::Setting &cfgLattice, Array<int> &unitCellTypes, Array2D<double> &unitCellPositions, 
  std::map<std::string,int> &atomTypeMap, Array4D<int> &latt, std::vector<int> &atom_type, std::vector<int> &type_count, 
  const std::vector<int> &dim, const int nAtoms, bool pbc[3]) {
  using namespace globals;
  const int maxatoms = dim[0]*dim[1]*dim[2]*nAtoms;
  assert(maxatoms > 0);

  latt.resize(dim[0],dim[1],dim[2],nAtoms);

  for (int x=0; x<dim[0]; ++x) {
    for (int y=0; y<dim[1]; ++y) {
      for (int z=0; z<dim[2]; ++z) {
        for (int n=0; n<nAtoms; ++n) {
          latt(x,y,z,n) = -1;
        }
      }
    }
  }
  
  
  atom_type.reserve(maxatoms);

  std::string shape;

  if(cfgLattice.lookupValue("shape",shape)) {
    std::transform(shape.begin(),shape.end(),shape.begin(),toupper);
    if( pbc[0] || pbc[1] || pbc[2] ) {
      output.write("\n************************************************************************\n");
      output.write("WARNING: Periodic boundaries and shape function applied\n");
      output.write("************************************************************************\n\n");
    }
  } else {
    shape = "DEFAULT";
    output.write("No shape function give\n");
  }
    
  int counter = 0;

  if(shape == "DEFAULT") {
    for (int x=0; x<dim[0]; ++x) {
      for (int y=0; y<dim[1]; ++y) {
        for (int z=0; z<dim[2]; ++z) {
          for (int n=0; n<nAtoms; ++n) {
            const int typeNum = unitCellTypes(n);
            atom_type.push_back(typeNum);
            type_count[typeNum]++;
            latt(x,y,z,n) = counter++;
          } // n
        } // z
      } // y
    } // x

  }
  else if(shape == "ELLIPSOID") {
    for (int x=0; x<dim[0]; ++x) {
      for (int y=0; y<dim[1]; ++y) {
        for (int z=0; z<dim[2]; ++z) {
          const double a = 0.5*dim[0];
          const double b = 0.5*dim[1];
          const double c = 0.5*dim[2];
          
          if( ((x-a)*(x-a)/(a*a) + (y-b)*(y-b)/(b*b) + (z-c)*(z-c)/(c*c)) < 1.0) {

            for (int n=0; n<nAtoms; ++n) {

              const int typeNum = unitCellTypes(n);
              atom_type.push_back(typeNum);
              type_count[typeNum]++;
              latt(x,y,z,n) = counter++;
            } // n
          }
        } // z
      } // y
    } // x
  }
  else {
    jams_error("Unknown shape function: %s\n",shape.c_str());
  }

  nspins = counter;
  nspins3 = 3*nspins;

  output.write("Total atoms: %i\n",nspins);
}

///
/// @brief  Print lattice to file.
///
void printLattice(const Array<int> &unitCellTypes, const Array2D<double> &unitCellPositions, Array4D<int> &latt, std::vector<int> &dim, const double unitcell[3][3], const int nAtoms) {
  using namespace globals;
  assert(nspins > 0);

  std::ofstream structfile;
  structfile.open("structure.out");
  structfile << nspins << "\n";
  structfile << "Generated by JAMS++\n";
  
  double r[3], p[3];
  int q[3];
  for (int x=0; x<dim[0]; ++x) {
    for (int y=0; y<dim[1]; ++y) {
      for (int z=0; z<dim[2]; ++z) {
        for (int n=0; n<nAtoms; ++n) {
          if(latt(x,y,z,n) != -1) {
            structfile << unitCellTypes(n) <<"\t";
            q[0] = x; q[1] = y; q[2] = z;
            for(int i=0; i<3; ++i) {
              r[i] = 0.0;
              p[i] = unitCellPositions(n,i);
              for(int j=0; j<3; ++j) {
                r[i] += unitcell[j][i]*(q[j]+p[i]);
              }
            }
            structfile << 5*r[0] <<"\t"<< 5*r[1] <<"\t"<< 5*r[2] <<"\n";
          }
        } // n
      } // z
    } // y
  } // x
  structfile.close();

}

///
/// @brief  Resize global arrays.
///
void resizeGlobals() {
  using namespace globals;
  assert(nspins > 0);
  s.resize(nspins,3);
  h.resize(nspins,3);
  w.resize(nspins,3);
  alpha.resize(nspins);
  mus.resize(nspins);
  gyro.resize(nspins);
  omega_corr.resize(nspins);
}

///
/// @brief  Initialise global arrays with material parameters.
///
void initialiseGlobals(libconfig::Config &config, const libconfig::Setting &cfgMaterials, std::vector<int> &atom_type) {
  using namespace globals;

    for(int i=0; i<nspins; ++i) {
      int type_num = atom_type[i];
      double sinit[3];
      double norm=0.0;
      for(int j=0;j<3;++j) {
        sinit[j] = cfgMaterials[type_num]["spin"][j]; 
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

      mus(i) = cfgMaterials[type_num]["moment"];
      mus(i) = mus(i);//*mu_bohr_si;
      
      alpha(i) = cfgMaterials[type_num]["alpha"];

      gyro(i) = cfgMaterials[type_num]["gyro"];
      gyro(i) = -gyro(i)/((1.0+alpha(i)*alpha(i))*mus(i));
    }
}

///
/// @brief  Read the interaction parameters from configuration file.
///
void readInteractions(libconfig::Config &config, const Array<int> &unitCellTypes, const Array2D<double> &unitCellPositions, Array3D<double> &interactionVectors, 
  Array2D<int> &interactionNeighbour, Array4D<double> &interactionValues, std::vector<int> &nInteractionsOfType, const int nAtoms, std::map<std::string,int> &atomTypeMap, const double unitcellInv[3][3]) {
  using namespace globals;
    
  int nInterTotal=0;

  // THIS NEEDS TO BE PASSED IN
  bool jsym = config.lookup("lattice.jsym");

  int nInterConfig = 0;

  double r[3],p[3];

  std::vector<int> atomInterCount(nspins,0);

  std::ifstream exchangeFile("exch.in");

  // count number of interactions
  if(jsym == true) {
    int atom1=0;
    int atom2=0;

    if( exchangeFile.is_open() ) {
      int n=0;
      for( std::string line; getline(exchangeFile,line); ) {
        std::stringstream is(line);

        is >> atom1;
        is >> atom2;

        is >> r[0];
        is >> r[1];
        is >> r[2];
      
        std::sort(r,r+3);
        do {
          output.write("%d: %f %f %f\n",n,r[0],r[1],r[2]);

          // count number of interaction for each atom in unit cell
          atomInterCount[atom1-1]++;

          nInterTotal++;
        } while (next_point_symmetry(r));
        n++;
        nInterConfig++;
      }
    }
  } else {

    if( exchangeFile.is_open() ) {
      int atom1=0;
      for( std::string line; getline(exchangeFile,line); ) {
        std::stringstream is(line);

        is >> atom1;

        // count number of interaction for each atom in unit cell
        atomInterCount[atom1-1]++;

        nInterTotal++;
        nInterConfig++;
      }
    }
  }


  // find maximum number of exchanges for a given atom in unit cell
  int interMax = 0;
  for(int i=0; i<nspins; ++i) {
    if(atomInterCount[i] > interMax) {
      interMax = atomInterCount[i];
    }
  }

  output.write("Interactions in config: %d\n",nInterConfig);
  output.write("Total interactions (with symmetry): %d\n",nInterTotal);


  // Guess number of interactions to minimise vector reallocing
  // int the sparse matrix inserts
  int inter_guess = 3*nspins*interMax;
  
  // Find number of exchange tensor components specified in the
  // config
  
  // rewind file
  exchangeFile.clear();
  exchangeFile.seekg(0,std::ios::beg);

  int nInteractions=0;
  if(nInterTotal > 0) {
    std::string line;
    int atom1=0;
    int atom2=0;

    getline(exchangeFile,line);
    std::stringstream is(line);

    is >> atom1;
    is >> atom2;

    is >> r[0];
    is >> r[1];
    is >> r[2];

    double tmp;
    while ( is >> tmp ) {
      nInteractions++;
    }
  } else {
    nInteractions = 0;
  }

  switch (nInteractions) {
    case 0:
      output.write("\n************************************************************************\n");
      output.write("WARNING: No exchange found\n");
      output.write("************************************************************************\n\n");
      break;
    case 1:
      output.write("Found isotropic exchange\n");
      break;
    case 2:
      output.write("Found uniaxial exchange\n");
      break;
    case 3:
      output.write("Found anisotropic exchange\n");
      break;
    case 9:
      output.write("Found tensorial exchange\n");
      inter_guess = nspins*interMax*9;
      break;
    default:
      jams_error("Undefined exchange symmetry. 1, 2, 3 or 9 components must be specified\n");
  }
  
  // Resize interaction arrays
  interactionVectors.resize(nAtoms,interMax,3);
  interactionNeighbour.resize(nAtoms,interMax);
  nInteractionsOfType.resize(nAtoms,0);

  // Resize global Jij matrix
  Jij.resize(nspins3,nspins3,inter_guess);

  //-----------------------------------------------------------------
  //  Read exchange tensor values from config
  //-----------------------------------------------------------------
  interactionValues.resize(nAtoms,interMax,3,3);

  // zero jij array
  for(int i=0; i<nAtoms; ++i) {
    for(int j=0; j<interMax; ++j) {
      for(int k=0; k<3; ++k) {
        for(int l=0; l<3; ++l) {
          interactionValues(i,j,k,l) = 0.0;
        }
      }
    }
  }


  // rewind file
  exchangeFile.clear();
  exchangeFile.seekg(0,std::ios::beg);

  int inter_counter = 0;
  for(int n=0; n<nInterConfig; ++n) {
    std::string line;
    
    getline(exchangeFile,line);
    std::stringstream is(line);

    double d_latt[3]={0.0,0.0,0.0};
    // read exchange tensor

    int atom_num_1=0;
    int atom_num_2=0;

    is >> atom_num_1;
    is >> atom_num_2;

    // count from zero
    atom_num_1--; atom_num_2--;

    for(int i=0; i<3; ++i) {
      // fractional vector within unit cell 
      p[i] = unitCellPositions(atom_num_2,i);
      // real space vector to neighbour
      is >> r[i];
    }

    if(jsym==true) {
      std::sort(r,r+3);
      do {
        // place interaction vector in unitcell space
        for(int i=0; i<3; ++i) {
          d_latt[i] = 0.0;
          for(int j=0; j<3; ++j) {
            d_latt[i] += r[j]*unitcellInv[j][i];
          }
        }

        // store unitcell atom difference
        interactionNeighbour(atom_num_1,nInteractionsOfType[atom_num_1]) = atom_num_2 - atom_num_1;

        // store interaction vectors
        for(int i=0;i<3; ++i){
          interactionVectors(atom_num_1,nInteractionsOfType[atom_num_1],i) = d_latt[i];
        }
  

        // read tensor components
        double J0=0.0,J1=0.0,J2=0.0;
        switch (nInteractions) {
          case 0:
            output.write("\n************************************************************************\n");
            output.write("WARNING: Attempting to insert non existent exchange");
            output.write("************************************************************************\n\n");
            break;
          case 1:
            is >> J0;
            interactionValues(atom_num_1,nInteractionsOfType[atom_num_1],0,0) = J0/mu_bohr_si; // Jxx
            interactionValues(atom_num_1,nInteractionsOfType[atom_num_1],1,1) = J0/mu_bohr_si; // Jyy
            interactionValues(atom_num_1,nInteractionsOfType[atom_num_1],2,2) = J0/mu_bohr_si; // Jzz
            break;
          case 2:
            is >> J0;
            is >> J1;
            interactionValues(atom_num_1,nInteractionsOfType[atom_num_1],0,0) = J0/mu_bohr_si; // Jxx
            interactionValues(atom_num_1,nInteractionsOfType[atom_num_1],1,1) = J0/mu_bohr_si; // Jyy
            interactionValues(atom_num_1,nInteractionsOfType[atom_num_1],2,2) = J1/mu_bohr_si; // Jzz
            break;
          case 3:
            is >> J0;
            is >> J1;
            is >> J2;
            interactionValues(atom_num_1,nInteractionsOfType[atom_num_1],0,0) = J0/mu_bohr_si; // Jxx
            interactionValues(atom_num_1,nInteractionsOfType[atom_num_1],1,1) = J1/mu_bohr_si; // Jyy
            interactionValues(atom_num_1,nInteractionsOfType[atom_num_1],2,2) = J2/mu_bohr_si; // Jzz
            break;
          case 9:
            for(int i=0; i<3; ++i) {
              for(int j=0; j<3; ++j) {
                is >> J0;
                interactionValues(atom_num_1,nInteractionsOfType[atom_num_1],i,j) = J0/mu_bohr_si;
              }
            }
            break;
          default:
            jams_error("Undefined exchange symmetry. 1, 2, 3 or 9 components must be specified\n");
        }

        nInteractionsOfType[atom_num_1]++;
        inter_counter++;
      } while (next_point_symmetry(r));
    } else {
      // place interaction vector in unitcell space
      matmul(unitcellInv,r,d_latt);
      
      // store unitcell atom difference
      interactionNeighbour(atom_num_1,nInteractionsOfType[atom_num_1]) = atom_num_2 - atom_num_1;
      
      // store interaction vectors
      for(int i=0;i<3; ++i){
        interactionVectors(atom_num_1,nInteractionsOfType[atom_num_1],i) = d_latt[i];
      }
      //std::cerr<<n<<"\t"<<atom_num_1<<"\t"<<atom_num_2<<"\t d_latt: "<<d_latt[0]<<"\t"<<d_latt[1]<<"\t"<<d_latt[2]<<std::endl;

        
      // read tensor components
      double J0=0.0,J1=0.0,J2=0.0;

      switch (nInteractions) {
        case 0:
          output.write("\n************************************************************************\n");
          output.write("WARNING: Attempting to insert non existent exchange");
          output.write("************************************************************************\n\n");
          break;
        case 1:
          is >> J0;
          interactionValues(atom_num_1,nInteractionsOfType[atom_num_1],0,0) = J0/mu_bohr_si; // Jxx
          interactionValues(atom_num_1,nInteractionsOfType[atom_num_1],1,1) = J0/mu_bohr_si; // Jyy
          interactionValues(atom_num_1,nInteractionsOfType[atom_num_1],2,2) = J0/mu_bohr_si; // Jzz
          break;
        case 2:
          is >> J0;
          is >> J1;
          interactionValues(atom_num_1,nInteractionsOfType[atom_num_1],0,0) = J0/mu_bohr_si; // Jxx
          interactionValues(atom_num_1,nInteractionsOfType[atom_num_1],1,1) = J0/mu_bohr_si; // Jyy
          interactionValues(atom_num_1,nInteractionsOfType[atom_num_1],2,2) = J1/mu_bohr_si; // Jzz
          break;
        case 3:
          is >> J0;
          is >> J1;
          is >> J2;
          interactionValues(atom_num_1,nInteractionsOfType[atom_num_1],0,0) = J0/mu_bohr_si; // Jxx
          interactionValues(atom_num_1,nInteractionsOfType[atom_num_1],1,1) = J1/mu_bohr_si; // Jyy
          interactionValues(atom_num_1,nInteractionsOfType[atom_num_1],2,2) = J2/mu_bohr_si; // Jzz
          break;
        case 9:
          for(int i=0; i<3; ++i) {
            for(int j=0; j<3; ++j) {
              is >> J0;
              interactionValues(atom_num_1,nInteractionsOfType[atom_num_1],i,j) = J0/mu_bohr_si;
            }
          }
          break;
        default:
          jams_error("Undefined exchange symmetry. 1, 2, 3 or 9 components must be specified\n");
      }
        
      inter_counter++;
      nInteractionsOfType[atom_num_1]++;
    }

  }
}

///
/// @brief  Create interaction matrix.
///
void createInteractionMatrix(libconfig::Config &config, const libconfig::Setting &cfgMaterials, Array4D<int> &latt, 
  const std::vector<int> dim, const int nAtoms, const Array<int> &unitCellTypes, const Array2D<double> &unitCellPositions, const std::vector<int> &atom_type, const Array3D<double> &interactionVectors,
  const Array2D<int> &interactionNeighbour, const Array4D<double> &interactionValues, const std::vector<int> &nInteractionsOfType, 
  const double unitcellInv[3][3], const bool pbc[3]) 
{
  
  using namespace globals;

  // single row of interaction matrix
  Array3D<double> spinInteractions(nspins,3,3);

  //double encut = 1e-25;  // energy cutoff
  double p[3], pnbr[3];
  int v[3], q[3], qnbr[3];

  bool surfaceAnisotropy=false;
  if(config.exists("lattice.surfaceAnisotropy")){
    surfaceAnisotropy = true;
    output.write("Neel surface anisotropy on\n");
  }else{
    surfaceAnisotropy = false;
  }
  double surfaceAnisotropyValue = 0.0;
  int surfaceCount = 0;

  int counter = 0;
  for (int x=0; x<dim[0]; ++x) {
    for (int y=0; y<dim[1]; ++y) {
      for (int z=0; z<dim[2]; ++z) {
        for (int n=0; n<nAtoms; ++n) {

          for(int i=0; i<nspins; ++i) {
            for(int j=0; j<3; ++j) {
              for(int k=0; k<3; ++k) {
                spinInteractions(i,j,k) = 0.0;
              }
            }
          }

          if(latt(x,y,z,n) != -1) {
            const int s_i = latt(x,y,z,n);
            const int type_num = atom_type[s_i];

            assert(s_i < nspins);
            
            // TODO: Check for if this is set
            if(surfaceAnisotropy == true) {
              surfaceAnisotropyValue = cfgMaterials[type_num]["surfaceAnisotropy"];
              surfaceAnisotropyValue = surfaceAnisotropyValue/mu_bohr_si;
            }
            
            q[0] = x; q[1] = y; q[2] = z;
            
            for(int j=0; j<3; ++j) {
              p[j] = unitCellPositions(n,j);
            }
          


            int localInteractionCount = 0;
            for(int i=0; i<nInteractionsOfType[n]; ++i) {
              // neighbour atom number
              int m = (interactionNeighbour(n,i)+n)%nAtoms;

              assert(m >= 0);
              
              for(int j=0; j<3; ++j) {
                pnbr[j] =  unitCellPositions(m,j);
              }

              double pnbrcell[3]={0.0,0.0,0.0};
              // put pnbr in unit cell
              matmul(unitcellInv,pnbr,pnbrcell);

              for(int j=0; j<3; ++j) {
                qnbr[j] = floor(interactionVectors(n,i,j)-pnbrcell[j]+0.5);
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

              if(idxcheck == true && (latt(v[0],v[1],v[2],m) != -1) ) {
                int nbr = latt(v[0],v[1],v[2],m);
                assert(nbr < nspins);
                assert(s_i != nbr);

#ifndef CUDA    // cusparse does not support symmetric matrices
                if(s_i > nbr) { // store only lower triangle
#endif
                  for(int row=0; row<3; ++row) {
                    for(int col=0; col<3; ++col) {
                      spinInteractions(nbr,row,col) = interactionValues(n,i,row,col); 
                    }
                  }
#ifndef CUDA
                }
#endif
                localInteractionCount++;
                counter++;
                if(surfaceAnisotropy == true) {

                  double u[3]; // unitvector
                  double norm = 0.0;
                  for(int j=0; j<3; ++j){
                    u[j] = p[j]+interactionVectors(n,i,j);
                    norm += u[j]*u[j];
                  }

                  norm = 1.0/sqrt(norm);
                  for(int j=0; j<3; ++j){
                    u[j] = u[j]*norm;
                  }
//                   std::cout<<u[0]<<"\t"<<u[1]<<"\t"<<u[2]<<std::endl;


                  // neighbour unit vectors
                  for(int row=0; row<3; ++row) {
                    for(int col=0; col<3; ++col) {
#ifndef CUDA    // cusparse does not support symmetric matrices
                      if(row >= col) { // store only lower triangle
#endif
                        spinInteractions(s_i,row,col) += surfaceAnisotropyValue*u[row]*u[col];
#ifndef CUDA
                      }
#endif
                    }
                  }
                }
              }

            } // interactionsOfType



//             std::cout<<localInteractionCount<<"\t"<<nInteractionsOfType[n]<<std::endl;
            if( (surfaceAnisotropy == false) || (localInteractionCount == nInteractionsOfType[n])) {

              // remove surface anisotropy
              for(int row=0; row<3; ++row) {
                for(int col=0; col<3; ++col) {
                  spinInteractions(s_i,row,col) = 0.0;
                }
              }

              // Bulk anisotropy
              double anival = cfgMaterials[type_num]["uniaxialAnisotropy"][1];

              for(int i=0;i<3;++i) {
                // easy axis
                double ei = cfgMaterials[type_num]["uniaxialAnisotropy"][0][i];
                // magnitude
                double di = 2.0*anival*ei ; 
                spinInteractions(s_i,i,i) = di/mu_bohr_si;
              }
            } else {
              surfaceCount++;
            }

              const double encut = 1E-26/mu_bohr_si; // energy cutoff
              for(int row=0; row<3; ++row){
                for(int j=0; j<nspins; ++j) {
                  for(int col=0; col<3; ++col) {
                    const double value = spinInteractions(j,row,col);
                    if(fabs(value) > encut) {
                      Jij.insert(3*s_i+row,3*j+col,value);
//                     std::cerr<<3*s_i+row<<"\t"<<3*j+col<<"\t"<<value<<"\n";
                    }

                  }
                }
              }
            
          }
        } // n
      } // z
    } // y
  } // x


  if(surfaceAnisotropy == true) {
    output.write("\nSurface count: %i\n", surfaceCount);
  }
  output.write("\nInteraction count: %i\n", counter);
  output.write("Jij memory (COO): %f MB\n",Jij.memorySize());
  output.write("Converting COO to CSR INPLACE\n");
  Jij.coocsrInplace();
//     Jij.coocsr();
  output.write("Jij memory (CSR): %f MB\n",Jij.memorySize());
}



void Lattice::createFromConfig(libconfig::Config &config) {
  using namespace globals;

  try {

    {
    Array4D<int> latt;
    Array2D<double> unitCellPositions;
    Array<int>      unitCellTypes;
    Array3D<double> interactionVectors;
    Array4D<double> interactionValues;
    Array2D<int> interactionNeighbour;
    std::vector<int> nInteractionsOfType;
    Array3D<double> jij;
    int nAtoms=0;
    bool pbc[3] = {true,true,true};

    double unitcell[3][3];
    double unitcellInv[3][3];
  
  
    const libconfig::Setting& cfgLattice    =   config.lookup("lattice");
    const libconfig::Setting& cfgBasis      =   config.lookup("lattice.unitcell.basis");
    const libconfig::Setting& cfgMaterials  =   config.lookup("materials");
//    const libconfig::Setting& cfgExchange   =   config.lookup("exchange");


    readBasis(cfgBasis, unitcell, unitcellInv);

    readAtoms(unitCellTypes, unitCellPositions, nAtoms, nTypes, atomTypeMap);
    assert(nAtoms > 0);
    assert(nTypes > 0);

    readLattice(cfgLattice, dim, pbc);

    type_count.resize(nTypes);
    for(int i=0; i<nTypes; ++i) { type_count[i] = 0; }

    createLattice(cfgLattice,unitCellTypes, unitCellPositions, atomTypeMap,latt,atom_type,type_count,dim,nAtoms,pbc);

#ifdef DEBUG
    printLattice(unitCellTypes, unitCellPositions,latt,dim,unitcell,nAtoms);
#endif // DEBUG

    resizeGlobals();

    initialiseGlobals(config, cfgMaterials, atom_type);

    readInteractions(config, unitCellTypes, unitCellPositions,interactionVectors, interactionNeighbour, interactionValues, nInteractionsOfType, 
      nAtoms, atomTypeMap, unitcellInv);

    createInteractionMatrix(config, cfgMaterials,latt,dim, nAtoms, unitCellTypes, unitCellPositions,
      atom_type,interactionVectors, interactionNeighbour, interactionValues,
      nInteractionsOfType, unitcellInv,pbc);
    }

  } // try
  catch(const libconfig::SettingNotFoundException &nfex) {
    jams_error("Setting '%s' not found",nfex.getPath());
  }

}
