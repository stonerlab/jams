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

  output.write("\n    Lattice translation vectors\n    ---------------------------\n");
  for(int i=0; i<3; ++i) { 
    output.write("    %f %f %f\n",unitcell[i][0],unitcell[i][1],unitcell[i][2]); 
  }
  
  output.write("\n    Inverse lattice vectors\n    ---------------------------\n");
  for(int i=0; i<3; ++i) { 
    output.write("    %f %f %f\n",unitcellInv[i][0],unitcellInv[i][1],unitcellInv[i][2]); 
  }
}

///
/// @brief  Read atom positions and types from config file.
///
void readAtoms(std::string &positionFileName, Array<int> &unitCellTypes, Array2D<double> &unitCellPositions, int &nAtoms, int &nTypes, std::map<std::string,int> &atomTypeMap) {
  //  Read atomic positions and types from config
  

  std::ifstream positionFile(positionFileName.c_str());

  nAtoms = 0;
  for (std::string line; getline(positionFile,line);) {
    nAtoms++;
  }

  unitCellTypes.resize(nAtoms);
  unitCellPositions.resize(nAtoms,3);

  output.write("\n    Atoms in unit cell\n    ------------------\n");
  
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

    output.write("    %s %f %f %f\n",typeName.c_str(), unitCellPositions(n,0), unitCellPositions(n,1), unitCellPositions(n,2));

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
  output.write("\n  * Unique atom types found: %d\n",nTypes);

  positionFile.close();
}

///
/// @brief  Read lattice parameters from config file.
///
void Lattice::readLattice(const libconfig::Setting &cfgLattice, std::vector<int> &dim, bool pbc[3]) {
  //  Read lattice properties from config
  
  for(int i=0; i<3; ++i) { 
    dim[i] = cfgLattice["size"][i]; 
  }
  output.write("  * Lattice size: %i %i %i\n",dim[0],dim[1],dim[2]);

  output.write("  * Boundary conditions: ");

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

  if( config.exists("lattice.kpoints") == true ) {
      for(int i=0; i<3; ++i) {
          unitcell_kpoints[i] = cfgLattice["kpoints"][i];
      }
  } else {
      for(int i=0; i<3; ++i) {
          unitcell_kpoints[i] = 1;
      }
  }
    
  output.write("  * Kpoints per unit cell: %i %i %i\n",unitcell_kpoints[0],unitcell_kpoints[1],unitcell_kpoints[2]);
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
    output.write("  * NO shape function\n");
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

  output.write("  * Total atoms in lattice: %i\n",nspins);
}

void Lattice::calculateAtomPos(const Array<int> &unitCellTypes, const Array2D<double> &unitCellPositions, Array4D<int> &latt, std::vector<int> &dim, const double unitcell[3][3], const int nAtoms) {
  using namespace globals;
  assert(nspins > 0);

  atom_pos.resize(nspins,3);
  
  atom_pos.resize(nspins,3);

  double r[3], p[3];
  int q[3];
  int count = 0;
  for (int x=0; x<dim[0]; ++x) {
    for (int y=0; y<dim[1]; ++y) {
      for (int z=0; z<dim[2]; ++z) {
        for (int n=0; n<nAtoms; ++n) {
          if(latt(x,y,z,n) != -1) {
            q[0] = x; q[1] = y; q[2] = z;
            for(int i=0; i<3; ++i) {
              r[i] = 0.0;
              p[i] = unitCellPositions(n,i);
              r[i] = q[i] + p[i];
              //for(int j=0; j<3; ++j) {
                //r[i] += unitcell[j][i]*(q[j]+p[i]);
              //}
            }
            for(int i=0; i<3; ++i){
              atom_pos(count,i) = r[i];
            }
            count++;
          }
        } // n
      } // z
    } // y
  } // x
  assert(count == nspins);
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
            structfile << r[0] <<"\t"<< r[1] <<"\t"<< r[2] <<"\n";
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

  output.write("\nInitialising global variables...\n");
    for(int i=0; i<nspins; ++i) {
      int type_num = atom_type[i];
      bool spinrand = cfgMaterials[type_num]["spinRand"];
      if( spinrand == true){
          rng.sphere(s(i,0),s(i,1),s(i,2));

            for(int j=0;j<3;++j){
                h(i,j) = 0.0;
                w(i,j) = 0.0;
            }
      }else{
        for(int j=0;j<3;++j) {
          s(i,j) = cfgMaterials[type_num]["spin"][j]; 
        }
        double norm = sqrt(s(i,0)*s(i,0) + s(i,1)*s(i,1) + s(i,2)*s(i,2));
      
        for(int j=0;j<3;++j){
            s(i,j) = s(i,j)/norm;
            h(i,j) = 0.0;
            w(i,j) = 0.0;
        }
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
/// @brief  Read the fourspin interaction parameters from configuration file.
///
void readJ4Interactions(std::string &J4FileName, libconfig::Config &config, const Array<int> &unitCellTypes, const Array2D<double> &unitCellPositions, Array4D<double> &J4Vectors, 
  Array3D<int> &J4Neighbour, Array2D<double> &J4Values, std::vector<int> &nJ4InteractionsOfType, const int nAtoms, std::map<std::string,int> &atomTypeMap, const double unitcellInv[3][3], int &nJ4Values) {
  using namespace globals;

  output.write("\nReading fourspin interaction file...\n");

  double Jval;
  int nInterTotal=0;
  int nInterConfig = 0;
  double r[3],p[3];
  std::vector<int> atomInterCount(nspins,0);

  std::ifstream exchangeFile(J4FileName.c_str());

  // count number of interactions
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


  // find maximum number of exchanges for a given atom in unit cell
  int interMax = 0;
  for(int i=0; i<nspins; ++i) {
    if(atomInterCount[i] > interMax) {
      interMax = atomInterCount[i];
    }
  }

  output.write("  * Fourspin interactions in file: %d\n",nInterConfig);

  // Resize interaction arrays
  J4Vectors.resize(nAtoms,interMax,3,3);
  J4Neighbour.resize(nAtoms,interMax,3);
  nJ4InteractionsOfType.resize(nAtoms,0);

  //-----------------------------------------------------------------
  //  Read exchange tensor values from config
  //-----------------------------------------------------------------
  J4Values.resize(nAtoms,interMax);

  // zero jij array
  for(int i=0; i<nAtoms; ++i) {
    for(int j=0; j<interMax; ++j) {
      J4Values(i,j) = 0.0;
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

    // read exchange tensor

    double vij[3];

    int atom_num[4];

    for(int j=0;j<4;++j){
      is >> atom_num[j];
    }

    // count from zero
    for(int j=0;j<4;++j){
      atom_num[j]--;
    }

    // --------------- vij ----------------
    for(int j=1;j<4;++j){
      for(int i=0; i<3; ++i) {
        p[i] = unitCellPositions(atom_num[j],i);   // fractional vector within unit cell 
        is >> r[i];                               // real space vector to neighbour
      }
      matmul(unitcellInv,r,vij);                  // place interaction vector in unitcell space
      
      for(int i=0; i<3; ++i) {
        J4Neighbour(atom_num[0],nJ4InteractionsOfType[atom_num[0]],j-1) = atom_num[j] - atom_num[0]; // store unitcell atom difference
        
        J4Vectors(atom_num[0],nJ4InteractionsOfType[atom_num[0]],j-1,i) = vij[i];
      }
    }
    
    is >> Jval; // bilinear
    J4Values(atom_num[0],nJ4InteractionsOfType[atom_num[0]]) = Jval/mu_bohr_si; // Jxx Jyy Jzz
      
    inter_counter++;
    nJ4InteractionsOfType[atom_num[0]]++;
  }
  


}
///
/// @brief  Read the interaction parameters from configuration file.
///
void readInteractions(std::string &exchangeFileName, libconfig::Config &config, const Array<int> &unitCellTypes, const Array2D<double> &unitCellPositions, Array3D<double> &interactionVectors, 
  Array2D<int> &interactionNeighbour, Array4D<double> &JValues, Array2D<double> &J2Values, std::vector<int> &nInteractionsOfType, const int nAtoms, std::map<std::string,int> &atomTypeMap, const double unitcellInv[3][3], bool &J2Toggle, int &nJValues) {
  using namespace globals;
  
  output.write("\nReading interaction file...\n");
  
  if( !config.lookupValue("lattice.biquadratic",J2Toggle) ) {
    J2Toggle = false;
  }else if (!J2Toggle){
    output.write("  * Biquadratic exchange ON\n");
    output.write("\n************************************************************************\n");
    output.write("Biquadratic values will be read from the last column of the exchange file\n");
    output.write("************************************************************************\n\n");
  }

  int nInterTotal=0;

  // THIS NEEDS TO BE PASSED IN
  bool jsym = config.lookup("lattice.jsym");

  int nInterConfig = 0;

  double r[3],p[3];

  std::vector<int> atomInterCount(nspins,0);

  std::ifstream exchangeFile(exchangeFileName.c_str());

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

  output.write("  * Interactions in file: %d\n",nInterConfig);
  output.write("  * Total interactions (with symmetry): %d\n",nInterTotal);


  // Find number of exchange tensor components specified in the
  // config
  
  // rewind file
  exchangeFile.clear();
  exchangeFile.seekg(0,std::ios::beg);

  nJValues=0;
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
      nJValues++;
    }
  } else {
    nJValues = 0;
  }

  // remove extra count for last column if biquadratic is present
  if(J2Toggle == true){
    nJValues--;
  }

  switch (nJValues) {
    case 0:
      output.write("\n************************************************************************\n");
      output.write("WARNING: No bilinear exchange found\n");
      output.write("************************************************************************\n\n");
      break;
    case 1:
      output.write("  * Isotropic exchange (1 component)\n");
      break;
    case 2:
      output.write("\tUniaxial exchange (2 components)\n");
      break;
    case 3:
      output.write("\tAnisotropic exchange (3 components)\n");
      break;
    case 9:
      output.write("\tTensorial exchange (9 components)\n");
      break;
    default:
      jams_error("Undefined exchange symmetry. 1, 2, 3 or 9 components must be specified\n");
  }
  
  // Resize interaction arrays
  interactionVectors.resize(nAtoms,interMax,3);
  interactionNeighbour.resize(nAtoms,interMax);
  nInteractionsOfType.resize(nAtoms,0);

  // Resize global J1ij_t and J2ij_t matrices
  J1ij_s.resize(nspins,nspins);
  J1ij_t.resize(nspins3,nspins3);
  J2ij_t.resize(nspins3,nspins3);


  if( J2Toggle == true ){
    // Resize biquadratic matrix
    // NOTE: this matrix is NxN because we use a custom routine so the
    // matrix is just a convenient neighbour list.
    J2ij_s.resize(nspins,nspins);
    J2ij_s.setMatrixType(SPARSE_MATRIX_TYPE_SYMMETRIC);
    J2ij_s.setMatrixMode(SPARSE_MATRIX_MODE_LOWER);
  }

  // Set matrix types
  std::string solname;
  if( config.exists("sim.solver") == true ) {
    config.lookupValue("sim.solver",solname);
    std::transform(solname.begin(),solname.end(),solname.begin(),toupper);
  } else {
    solname = "DEFAULT";
  }
  if( ( solname == "CUDAHEUNLLG" ) || ( solname == "CUDASEMILLG" ) ) {
    output.write("  * CUDA solver means a symmetric (lower) sparse matrix will be stored\n");
    J1ij_s.setMatrixType(SPARSE_MATRIX_TYPE_SYMMETRIC);
    J1ij_s.setMatrixMode(SPARSE_MATRIX_MODE_LOWER);
    J1ij_t.setMatrixType(SPARSE_MATRIX_TYPE_SYMMETRIC);
    J1ij_t.setMatrixMode(SPARSE_MATRIX_MODE_LOWER);
    J2ij_t.setMatrixType(SPARSE_MATRIX_TYPE_SYMMETRIC);
    J2ij_t.setMatrixMode(SPARSE_MATRIX_MODE_LOWER);
  } else {
    output.write("  * Symmetric (lower) sparse matrix will be stored\n");
    J1ij_s.setMatrixType(SPARSE_MATRIX_TYPE_SYMMETRIC);
    J1ij_s.setMatrixMode(SPARSE_MATRIX_MODE_LOWER);
    J1ij_t.setMatrixType(SPARSE_MATRIX_TYPE_SYMMETRIC);
    J1ij_t.setMatrixMode(SPARSE_MATRIX_MODE_LOWER);
    J2ij_t.setMatrixType(SPARSE_MATRIX_TYPE_SYMMETRIC);
    J2ij_t.setMatrixMode(SPARSE_MATRIX_MODE_LOWER);
  }

  //-----------------------------------------------------------------
  //  Read exchange tensor values from config
  //-----------------------------------------------------------------
  JValues.resize(nAtoms,interMax,3,3);
  J2Values.resize(nAtoms,interMax);

  // zero jij array
  for(int i=0; i<nAtoms; ++i) {
    for(int j=0; j<interMax; ++j) {
      J2Values(i,j) = 0.0;
      for(int k=0; k<3; ++k) {
        for(int l=0; l<3; ++l) {
          JValues(i,j,k,l) = 0.0;
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
        double Jval = 0.0;
        switch (nJValues) {
          case 0:
            if(J2Toggle == false){
              output.write("\n************************************************************************\n");
              output.write("WARNING: Attempting to insert non existent exchange");
              output.write("************************************************************************\n\n");
            }
            break;
          case 1:
            is >> Jval; // bilinear
            for(int i=0; i<3; ++i){
              JValues(atom_num_1,nInteractionsOfType[atom_num_1],i,i) = Jval/mu_bohr_si; // Jxx Jyy Jzz
            }
            break;
          case 2:
            is >> Jval;
            for(int i=0; i<2; ++i){
              JValues(atom_num_1,nInteractionsOfType[atom_num_1],i,i) = Jval/mu_bohr_si; // Jxx Jyy
            }
            is >> Jval;
            JValues(atom_num_1,nInteractionsOfType[atom_num_1],2,2) = Jval/mu_bohr_si; // Jzz
            break;
          case 3:
            for(int i=0; i<3; ++i){
              is >> Jval;
              JValues(atom_num_1,nInteractionsOfType[atom_num_1],i,i) = Jval/mu_bohr_si; // Jxx Jyy Jzz
            }
            break;
          case 9:
            for(int i=0; i<3; ++i) {
              for(int j=0; j<3; ++j) {
                is >> Jval;
                JValues(atom_num_1,nInteractionsOfType[atom_num_1],i,j) = Jval/mu_bohr_si;
              }
            }
            break;
          default:
            jams_error("Undefined exchange symmetry. 1, 2, 3 or 9 components must be specified\n");
        }

        // extra column at the end if biquadratic is on
        if(J2Toggle == true){
          is >> Jval;
          J2Values(atom_num_1,nInteractionsOfType[atom_num_1]) = Jval/mu_bohr_si;
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
      double Jval = 0.0;
      switch (nJValues) {
        case 0:
          if(J2Toggle == false){
            output.write("\n************************************************************************\n");
            output.write("WARNING: Attempting to insert non existent exchange");
            output.write("************************************************************************\n\n");
          }
          break;
        case 1:
          is >> Jval; // bilinear
          for(int i=0; i<3; ++i){
            JValues(atom_num_1,nInteractionsOfType[atom_num_1],i,i) = Jval/mu_bohr_si; // Jxx Jyy Jzz
          }
          break;
        case 2:
          is >> Jval;
          for(int i=0; i<2; ++i){
            JValues(atom_num_1,nInteractionsOfType[atom_num_1],i,i) = Jval/mu_bohr_si; // Jxx Jyy
          }
          is >> Jval;
          JValues(atom_num_1,nInteractionsOfType[atom_num_1],2,2) = Jval/mu_bohr_si; // Jzz
          break;
        case 3:
          for(int i=0; i<3; ++i){
            is >> Jval;
            JValues(atom_num_1,nInteractionsOfType[atom_num_1],i,i) = Jval/mu_bohr_si; // Jxx Jyy Jzz
          }
          break;
        case 9:
          for(int i=0; i<3; ++i) {
            for(int j=0; j<3; ++j) {
              is >> Jval;
              JValues(atom_num_1,nInteractionsOfType[atom_num_1],i,j) = Jval/mu_bohr_si;
            }
          }
          break;
        default:
          jams_error("Undefined exchange symmetry. 1, 2, 3 or 9 components must be specified\n");
      }
        
      // extra column at the end if biquadratic is on
      if(J2Toggle == true){
        is >> Jval;
        J2Values(atom_num_1,nInteractionsOfType[atom_num_1]) = Jval/mu_bohr_si;
      }
        
      inter_counter++;
      nInteractionsOfType[atom_num_1]++;
    }

  }
}



///
/// @brief  Create J4 interaction matrix.
///
void createJ4Matrix(libconfig::Config &config, const libconfig::Setting &cfgMaterials, Array4D<int> &latt, 
  const std::vector<int> dim, const int nAtoms, const Array<int> &unitCellTypes, const Array2D<double> &unitCellPositions, const std::vector<int> &atom_type, const Array4D<double> &J4Vectors,
  const Array3D<int> &J4Neighbour, const Array2D<double> &J4Values, const std::vector<int> &nJ4InteractionsOfType, 
  const double unitcellInv[3][3], const bool pbc[3], const int &nJ4Values) 
{
  
  using namespace globals;
  output.write("\nCalculating fourspin interaction matrix...\n");
              
  const double encut = 1E-28/mu_bohr_si; // energy cutoff

  int qi[3], qj[3];
  double pi[3], pj[3];
  int vij[3];
  int sj[3];

  J4ijkl_s.resize(nspins,nspins,nspins,nspins);
  
  int counter = 0;
  for (int x=0; x<dim[0]; ++x) {
    for (int y=0; y<dim[1]; ++y) {
      for (int z=0; z<dim[2]; ++z) {
        for (int n=0; n<nAtoms; ++n) {

          if(latt(x,y,z,n) != -1) {

            const int si = latt(x,y,z,n); assert (si < nspins);

            qi[0] = x; qi[1] = y; qi[2] = z;

            for(int j=0; j<3; ++j){
              pi[j] = unitCellPositions(n,j);
            }
            
            for(int i=0; i<nJ4InteractionsOfType[n]; ++i) {

              // loop the 3 neighbours of the fourspin term
              for(int snbr=0; snbr<3; ++snbr){

                int m = (J4Neighbour(n,i,snbr)+n)%nAtoms; assert(m >=0);

                for(int j=0; j<3; ++j){
                  pj[j] = unitCellPositions(m,j); 
                }

                // put pj in unitcell
                double pj_cell[3] = {0.0,0.0,0.0};
                matmul(unitcellInv,pj,pj_cell);

                for(int j=0; j<3; ++j){
                  qj[j] = floor(J4Vectors(n,i,snbr,j) - pj_cell[j] + 0.5);
                }

                // calculate realspace interaction vector
                bool idxcheck = true;
                for(int j=0; j<3; ++j){
                  vij[j] = qi[j] - qj[j];
                  if(pbc[j] == true) {
                    vij[j] = (dim[j]+vij[j])%dim[j];
                  } else if (vij[j] < 0 || !(vij[j] < dim[j])) {
                    idxcheck = false;
                  }
                }

                // check interaction existed
                if(idxcheck == true) {
                  sj[snbr] = latt(vij[0],vij[1],vij[2],m); assert(sj < nspins);
                } else {
                  sj[snbr] = -1;
                }
              }


              if( (sj[0] > -1) && (sj[1] > -1) && (sj[2] > -1) ){
                if( fabs(J4Values(n,i)) > encut) {
                  J4ijkl_s.insertValue(si,sj[0],sj[1],sj[2],J4Values(n,i));
                }
                counter++;
              }


            }
          }
        }
      }
    }
  }

  output.write("  * Total J4 interaction count: %i\n", counter);
  output.write("  * J4ijkl_s matrix memory (MAP): %f MB\n",J4ijkl_s.calculateMemory());
}


///
/// @brief  Create interaction matrix.
///
void createInteractionMatrix(libconfig::Config &config, const libconfig::Setting &cfgMaterials, Array4D<int> &latt, 
  const std::vector<int> dim, const int nAtoms, const Array<int> &unitCellTypes, const Array2D<double> &unitCellPositions, const std::vector<int> &atom_type, const Array3D<double> &interactionVectors,
  const Array2D<int> &interactionNeighbour, const Array4D<double> &JValues, const std::vector<int> &nInteractionsOfType, 
  const double unitcellInv[3][3], const bool pbc[3],const bool &J2Toggle, const Array2D<double> &J2Values, const int &nJValues) 
{
  
  using namespace globals;
  output.write("\nCalculating interaction matrix...\n");
              
  const double encut = 1E-26/mu_bohr_si; // energy cutoff

  double p[3], pnbr[3];
  int v[3], q[3], qnbr[3];

  int counter = 0;
  for (int x=0; x<dim[0]; ++x) {
    for (int y=0; y<dim[1]; ++y) {
      for (int z=0; z<dim[2]; ++z) {
        for (int n=0; n<nAtoms; ++n) {

          if(latt(x,y,z,n) != -1) {

            const int s_i = latt(x,y,z,n);
            const int type_num = atom_type[s_i];

            assert(s_i < nspins);
            
            q[0] = x; q[1] = y; q[2] = z;
            
            for(int j=0; j<3; ++j) {
              p[j] = unitCellPositions(n,j);
            }
          


            int localInteractionCount = 0;
            for(int i=0; i<nInteractionsOfType[n]; ++i) {
              // neighbour atom number
              int m = (interactionNeighbour(n,i)+n)%nAtoms;

              assert(m >= 0);
              
              for(int j=0; j<3; ++j) { pnbr[j] =  unitCellPositions(m,j); }

              double pnbrcell[3]={0.0,0.0,0.0};
              // put pnbr in unit cell
              matmul(unitcellInv,pnbr,pnbrcell);

              for(int j=0; j<3; ++j) { qnbr[j] = floor(interactionVectors(n,i,j)-pnbrcell[j]+0.5); }

              bool idxcheck = true;
              for(int j=0; j<3; ++j) {
                v[j] = q[j]+qnbr[j];
                if(pbc[j] == true) {
                  v[j] = (dim[j]+v[j])%dim[j];
                } else if (v[j] < 0 || !(v[j] < dim[j])) {
                  idxcheck = false;
                }
              }

              if(idxcheck == true && (latt(v[0],v[1],v[2],m) != -1) ) {
                int nbr = latt(v[0],v[1],v[2],m);
                assert(nbr < nspins);
                //assert(s_i != nbr); self interaction is ok for example
                //FeRh

//---------------------------------------------------------------------
// Bilinear interactions
//---------------------------------------------------------------------
                if(nJValues == 1){
                //--------//
                // Scalar //
                //--------//
                  if(fabs(JValues(n,i,0,0)) > encut) {
                    if(J1ij_s.getMatrixType() == SPARSE_MATRIX_TYPE_SYMMETRIC) {
                      if(J1ij_s.getMatrixMode() == SPARSE_MATRIX_MODE_LOWER) {
                        if(s_i >= nbr){
                          J1ij_s.insertValue(s_i,nbr,JValues(n,i,0,0));
                        }
                      }else{
                        if(s_i <= nbr){
                          J1ij_s.insertValue(s_i,nbr,JValues(n,i,0,0));
                        }
                      }
                    }else{
                      J1ij_s.insertValue(s_i,nbr,JValues(n,i,0,0));
                    }
                  }
                } else {
                //--------//
                // Tensor //
                //--------//
                  for(int row=0; row<3; ++row) {
                    for(int col=0; col<3; ++col) {
                      if(fabs(JValues(n,i,row,col)) > encut) {
                        if(J1ij_t.getMatrixType() == SPARSE_MATRIX_TYPE_SYMMETRIC) {
                          if(J1ij_t.getMatrixMode() == SPARSE_MATRIX_MODE_LOWER) {
                            if(s_i > nbr){
                              J1ij_t.insertValue(3*s_i+row,3*nbr+col,JValues(n,i,row,col));
                            }
                          } else {
                            if(s_i < nbr){
                              J1ij_t.insertValue(3*s_i+row,3*nbr+col,JValues(n,i,row,col));
                            }
                          }
                        } else {
                          J1ij_t.insertValue(3*s_i+row,3*nbr+col,JValues(n,i,row,col));
                        }
                      }
                    }
                  }
                }

//---------------------------------------------------------------------
// Biquadratic interactions
//---------------------------------------------------------------------

                //--------//
                // Scalar //
                //--------//
                if(J2Toggle == true){
                  if(fabs(J2Values(n,i)) > encut) {
                    if(J2ij_s.getMatrixType() == SPARSE_MATRIX_TYPE_SYMMETRIC) {
                      if(J2ij_s.getMatrixMode() == SPARSE_MATRIX_MODE_LOWER) {
                        if(s_i >= nbr){
                          J2ij_s.insertValue(s_i,nbr,J2Values(n,i));
                        }
                      } else {
                        if(s_i <= nbr){
                          J2ij_s.insertValue(s_i,nbr,J2Values(n,i));
                        }
                      }
                    } else {
                      J2ij_s.insertValue(s_i,nbr,J2Values(n,i));
                    }
                  }
                }

                localInteractionCount++;
                counter++;
              }

            } 

//---------------------------------------------------------------------
// Anisotropy (Biquadratic Tensor)
//---------------------------------------------------------------------
            double anival = cfgMaterials[type_num]["uniaxialAnisotropy"][1];

            // NOTE: Technically anisotropic is biquadratic but
            // biquadratic interactions in JAMS++ are implemented as a
            // scalar not a tensor so we store these in a seperate
            // array

            // read anisotropy axis unit vector and ensure it is normalised
            double e3[3] = {0.0,0.0,0.0};
            for(int i=0;i<3;++i) {
              e3[i] = cfgMaterials[type_num]["uniaxialAnisotropy"][0][i];
            }

            double norm = 1.0/sqrt(e3[0]*e3[0]+e3[1]*e3[1]+e3[2]*e3[2]);
            
            for(int i=0;i<3;++i) {
              e3[i] = e3[i]*norm;
            }

            const double DTensor[3][3] = { { e3[0]*e3[0], e3[0]*e3[1], e3[0]*e3[2] },
                                           { e3[1]*e3[0], e3[1]*e3[1], e3[1]*e3[2] },
                                           { e3[2]*e3[0], e3[2]*e3[1], e3[2]*e3[2] } };

            // NOTE: Factor of 2 is accounted for in the biquadratic
            // calculation
            const double Dval = anival/mu_bohr_si;


            for(int row=0; row<3; ++row) {
              for(int col=0; col<3; ++col) {
                if((Dval*DTensor[row][col]) > encut) {
                  if(J2ij_t.getMatrixType() == SPARSE_MATRIX_TYPE_SYMMETRIC) {
                    if(J2ij_t.getMatrixMode() == SPARSE_MATRIX_MODE_LOWER) {
                      if(row >= col){
                        J2ij_t.insertValue(3*s_i+row,3*s_i+col,Dval*DTensor[row][col]);
                      }
                    } else {
                      if(row <= col){
                        J2ij_t.insertValue(3*s_i+row,3*s_i+col,Dval*DTensor[row][col]);
                      }
                    }
                  } else {
                    J2ij_t.insertValue(3*s_i+row,3*s_i+col,Dval*DTensor[row][col]);
                  }
                }
              }
            }
          }
        } // n
      } // z
    } // y
  } // x


  output.write("  * Total interaction count: %i\n", counter);
  output.write("  * J1ij_t matrix memory (MAP): %f MB\n",J1ij_t.calculateMemory());

// J1ij_t.printCSR();
}



void Lattice::createFromConfig(libconfig::Config &config) {
  using namespace globals;

  output.write("\nCalculating lattice...\n");

  try {

    {
    Array4D<int> latt;
    Array2D<double> unitCellPositions;
    Array<int>      unitCellTypes;
    Array3D<double> interactionVectors;
    Array4D<double> JValues;
    Array2D<double> J2Values;
    Array2D<int> interactionNeighbour;
    Array3D<int> J4Neighbour;
    Array4D<double> J4Vectors;
    Array2D<double> J4Values;
    std::vector<int> nInteractionsOfType;
    std::vector<int> nJ4InteractionsOfType;
    Array3D<double> jij;
    int nAtoms=0;
    bool pbc[3] = {true,true,true};
    bool J2Toggle = false;
    int nJValues=0;
    int nJ4Values=0;

    double unitcell[3][3];
    double unitcellInv[3][3];
  
  
    const libconfig::Setting& cfgLattice    =   config.lookup("lattice");
    const libconfig::Setting& cfgBasis      =   config.lookup("lattice.unitcell.basis");
    const libconfig::Setting& cfgMaterials  =   config.lookup("materials");
//    const libconfig::Setting& cfgExchange   =   config.lookup("exchange");


    readBasis(cfgBasis, unitcell, unitcellInv);

    std::string positionFileName = config.lookup("lattice.positions");
    readAtoms(positionFileName,unitCellTypes, unitCellPositions, nAtoms, nTypes, atomTypeMap);

    assert(nAtoms > 0);
    assert(nTypes > 0);

    readLattice(cfgLattice, dim, pbc);

    type_count.resize(nTypes);
    for(int i=0; i<nTypes; ++i) { type_count[i] = 0; }

    createLattice(cfgLattice,unitCellTypes, unitCellPositions, atomTypeMap,latt,atom_type,type_count,dim,nAtoms,pbc);

    calculateAtomPos(unitCellTypes, unitCellPositions,latt,dim,unitcell,nAtoms);
#ifdef DEBUG
    printLattice(unitCellTypes, unitCellPositions,latt,dim,unitcell,nAtoms);
#endif // DEBUG

    resizeGlobals();

    initialiseGlobals(config, cfgMaterials, atom_type);

    std::string exchangeFileName = config.lookup("lattice.exchange");
    readInteractions(exchangeFileName, config, unitCellTypes, unitCellPositions,interactionVectors, interactionNeighbour, JValues, J2Values, nInteractionsOfType, 
      nAtoms, atomTypeMap, unitcellInv, J2Toggle, nJValues);

    createInteractionMatrix(config, cfgMaterials,latt,dim, nAtoms, unitCellTypes, unitCellPositions,
      atom_type,interactionVectors, interactionNeighbour, JValues,
      nInteractionsOfType, unitcellInv,pbc,J2Toggle,J2Values,nJValues);
        
        mapPosToInt();
    
    if( config.exists("lattice.fourspin") == true ) {
      std::string J4FileName = config.lookup("lattice.fourspin");
      readJ4Interactions(J4FileName, config, unitCellTypes, unitCellPositions, J4Vectors, J4Neighbour, J4Values, nJ4InteractionsOfType, 
        nAtoms, atomTypeMap, unitcellInv, nJ4Values);

      createJ4Matrix(config, cfgMaterials,latt,dim, nAtoms, unitCellTypes, unitCellPositions,
        atom_type,J4Vectors, J4Neighbour, J4Values,
        nJ4InteractionsOfType, unitcellInv,pbc,nJ4Values);
      
    }
    }

  } // try
  catch(const libconfig::SettingNotFoundException &nfex) {
    jams_error("Setting '%s' not found",nfex.getPath());
  }

}

void Lattice::outputSpinsVTU(std::ofstream &outfile){
  using namespace globals;

  outfile << "<?xml version=\"1.0\"?>" << "\n"; 
  outfile << "<VTKFile type=\"UnstructuredGrid\">" << "\n";
  outfile << "<UnstructuredGrid>" << "\n";
  outfile << "<Piece NumberOfPoints=\""<<nspins<<"\"  NumberOfCells=\"1\">" << "\n";
  outfile << "<PointData Scalar=\"Spin\">" << "\n";
  outfile << "<DataArray type=\"Float32\" Name=\"Spin\" NumberOfComponents=\"3\" format=\"ascii\">" << "\n";
  for(int i=0; i<nspins; ++i){
    outfile << s(i,0) << "\t" << s(i,1) << "\t" << s(i,2) << "\n";
  }
  outfile << "</DataArray>" << "\n";
  outfile << "</PointData>" << "\n";
  outfile << "<CellData>" << "\n";
  outfile << "</CellData>" << "\n";
  outfile << "<Points>" << "\n";
  outfile << "<DataArray type=\"Float32\" NumberOfComponents=\"3\" format=\"ascii\">" << "\n";
  for(int i=0; i<nspins; ++i){
    outfile << atom_pos(i,0) << "\t" << atom_pos(i,1) << "\t" << atom_pos(i,2) << "\n";
  }
  outfile << "</DataArray>" << "\n";
  outfile << "</Points>" << "\n";
  outfile << "<Cells>" << "\n";
  outfile << "<DataArray type=\"Int32\" Name=\"connectivity\" format=\"ascii\">" << "\n";
  outfile << "1" << "\n";
  outfile << "</DataArray>" << "\n";
  outfile << "<DataArray type=\"Int32\" Name=\"offsets\" format=\"ascii\">" << "\n";
  outfile << "1" << "\n";
  outfile << "</DataArray>" << "\n";
  outfile << "<DataArray type=\"UInt8\" Name=\"types\" format=\"ascii\">" << "\n";
  outfile << "1" << "\n";
  outfile << "</DataArray>" << "\n";
  outfile << "</Cells>" << "\n";
  outfile << "</Piece>" << "\n";
  outfile << "</UnstructuredGrid>" << "\n";
  outfile << "</VTKFile>" << "\n";

}


void Lattice::mapPosToInt(){
    using namespace globals;

    spin_int_map.resize(nspins,3);

    for(int i=0; i<nspins; ++i){
        for(int j=0; j<3; ++j){
            spin_int_map(i,j) = nint(atom_pos(i,j)*unitcell_kpoints[j]);
        }
        //std::cout<<atom_pos(i,0)<<"\t"<<atom_pos(i,1)<<"\t"<<atom_pos(i,2)<<"\t";
        //std::cout<<spin_int_map(i,0)<<"\t"<<spin_int_map(i,1)<<"\t"<<spin_int_map(i,2)<<std::endl;
    }
}
