// Copyright 2014 Joseph Barker. All rights reserved.

#include "core/lattice.h"

#include <libconfig.h++>
#include <stdint.h>

#include <algorithm>
#include <cmath>
#include <fstream>
#include <iostream>
#include <map>
#include <sstream>
#include <string>
#include <utility>

#include "core/consts.h"
#include "core/globals.h"
#include "core/maths.h"
#include "core/sparsematrix.h"
#include "core/utils.h"

#include "jblib/containers/array.h"

///
/// @brief  Read basis vectors from config file.
///
void read_basis(const libconfig::Setting &cfgBasis,
  double unitcell[3][3], double unitcellInv[3][3]) {
  using namespace globals;

  // We transpose during the read because the unit cell matrix must have the
  // lattice vectors as the columns but it is easiest to define each vector in
  // the input
  //  / a1x a2x a2x \  / A \     / A.a1x + B.a2x + C.a3x \
  // |  a1y a2y a3y  ||  B  | = |  A.a1y + B.a2y + C.a3y  |
  //  \ a1z a2z a3z /  \ C /     \ A.a1z + B.a2z + C.a3z /
  for (int i = 0; i < 3; ++i) {
    for (int j = 0; j < 3; ++j) {
      unitcell[i][j] = cfgBasis[j][i];
    }
  }

  matrix_invert(unitcell, unitcellInv);

  output.write("\n    Lattice translation vectors");
  output.write("\n    ---------------------------\n");

  for (int i = 0; i < 3; ++i) {
    output.write("    %f %f %f\n",
      unitcell[i][0], unitcell[i][1], unitcell[i][2]);
  }

  output.write("\n    Inverse lattice vectors");
  output.write("\n    ---------------------------\n");
  for (int i = 0; i < 3; ++i) {
    output.write("    %f %f %f\n",
      unitcellInv[i][0], unitcellInv[i][1], unitcellInv[i][2]);
  }
}

///
/// @brief  Read atom positions and types from config file.
///
void readAtoms(std::string &atom_position_filename,
 jblib::Array<int, 1> &unit_cell_types,
 jblib::Array<double, 2> &unit_cell_positions,
 int &num_atoms, int &num_atom_types,
 std::map<std::string, int> &atom_type_map,
 std::vector<std::string> &atom_names) {
  std::ifstream atom_position_file(atom_position_filename.c_str());

  num_atoms = 0;
  for (std::string line; getline(atom_position_file, line);) {
    num_atoms++;
  }

  unit_cell_types.resize(num_atoms);
  unit_cell_positions.resize(num_atoms, 3);

  output.write("\n    Atoms in unit cell\n    ------------------\n");

  std::map<std::string, int>::iterator it_type;
  num_atom_types = 0;

  std::string atom_type_name;

  atom_position_file.clear();
  atom_position_file.seekg(0, std::ios::beg);

  for (int n = 0; n < num_atoms; ++n) {
    std::string line;

    getline(atom_position_file, line);

    std::stringstream is(line);

    is >> atom_type_name;
    for (int j = 0; j < 3; ++j) {
      is >> unit_cell_positions(n, j);
    }

#ifdef DEBUG
    output.write("    %s %f %f %f\n", atom_type_name.c_str(),
      unit_cell_positions(n, 0),
      unit_cell_positions(n, 1),
      unit_cell_positions(n, 2));
#else
    if (n < 8) {  // only print 8 atoms to avoid excessive output
      output.write("    %s %f %f %f\n", atom_type_name.c_str(),
        unit_cell_positions(n, 0),
        unit_cell_positions(n, 1),
        unit_cell_positions(n, 2));
    } else if ((n == 8) && (num_atoms > 8)) {
      output.write("    ...\n");
    }
    if (num_atoms > 8 && n == (num_atoms-1)) {
      output.write("    %s %f %f %f\n", atom_type_name.c_str(),
        unit_cell_positions(n, 0),
        unit_cell_positions(n, 1),
        unit_cell_positions(n, 2));
    }
#endif  // DEBUG

    it_type = atom_type_map.find(atom_type_name);
    if (it_type == atom_type_map.end()) {
      // type not found in map -> add to map
      // map type_name -> int
      unit_cell_types(n) = num_atom_types;

      atom_type_map.insert(std::pair<std::string, int>(atom_type_name, num_atom_types));
      atom_names.push_back(atom_type_name);
      num_atom_types++;
    } else {
      unit_cell_types(n) = atom_type_map[atom_type_name];
    }
  }
  output.write("\n  * Unique atom types found: %d\n", num_atom_types);

  atom_position_file.close();
}

///
/// @brief  Read lattice parameters from config file.
///
void Lattice::readLattice(const libconfig::Setting &cfgLattice,
  std::vector<int> &dim, bool pbc[3], const double unitcell[3][3]) {
  for (int i = 0; i < 3; ++i) {
    dim[i] = cfgLattice["size"][i];
  }
  output.write("  * Lattice size: %i %i %i\n", dim[0], dim[1], dim[2]);

  lattice_parameter = cfgLattice["parameter"];
  output.write("  * Lattice parameter: %f\n", lattice_parameter);

  for (int i = 0; i < 3; ++i) {
    rmax[i] = 0;
    for (int j = 0; j < 3; ++j) {
      rmax[i] += lattice_parameter*unitcell[i][j]*dim[j];
    }
  }

  output.write("  * Boundary conditions: ");
  for (int i = 0; i < 3; ++i) {
    pbc[i] = cfgLattice["periodic"][i];
    is_periodic[i] = pbc[i];
    if (pbc[i]) {
      output.write("periodic ");
    } else {
      output.write("open ");
    }
  }
  output.write("\n");

  if (config.exists("lattice.kpoints") == true) {
    for (int i = 0; i < 3; ++i) {
      unitcell_kpoints[i] = cfgLattice["kpoints"][i];
    }
  } else {
    jams_error("Number of kpoints not specified");
  }

  output.write("  * Kpoints per unit cell: %i %i %i\n",
    unitcell_kpoints[0], unitcell_kpoints[1], unitcell_kpoints[2]);
}

///
/// @brief  Create lattice on numbered spin locations.
///
void create_lattice(const libconfig::Setting &cfgLattice,
 jblib::Array<int, 1> &unit_cell_types,
 jblib::Array<double, 2> &unit_cell_positions,
 std::map<std::string, int> &atom_type_map,
 jblib::Array<int, 4> &latt,
 std::vector<int> &atom_type,
 std::vector<int> &type_count,
 const std::vector<int> &dim,
 const int num_atoms, bool pbc[3]) {
  using namespace globals;
  const int max_atoms_possible = dim[0]*dim[1]*dim[2]*num_atoms;
  assert(max_atoms_possible > 0);

  latt.resize(dim[0], dim[1], dim[2], num_atoms);

  for (int x = 0; x < dim[0]; ++x) {
    for (int y = 0; y < dim[1]; ++y) {
      for (int z = 0; z < dim[2]; ++z) {
        for (int n = 0; n < num_atoms; ++n) {
          latt(x, y, z, n) = -1;
        }
      }
    }
  }

  atom_type.reserve(max_atoms_possible);

  std::string lattice_shape_name;
  if (cfgLattice.lookupValue("lattice_shape_name", lattice_shape_name)) {
    lattice_shape_name = capitalize(lattice_shape_name);
    if( pbc[0] || pbc[1] || pbc[2] ) {
      output.write("\n************************************************************************\n");
      output.write("WARNING: Periodic is_periodic and lattice_shape_name function applied\n");
      output.write("************************************************************************\n\n");
    }
  } else {
    lattice_shape_name = "DEFAULT";
    output.write("  * NO lattice shape function\n");
  }

  int counter = 0;

  if(lattice_shape_name == "DEFAULT") {
    for (int x=0; x<dim[0]; ++x) {
      for (int y=0; y<dim[1]; ++y) {
        for (int z=0; z<dim[2]; ++z) {
          for (int n=0; n<num_atoms; ++n) {
            const int type_number = unit_cell_types(n);
            atom_type.push_back(type_number);
            type_count[type_number]++;
            latt(x, y, z, n) = counter++;
          } // n
        } // z
      } // y
    } // x

  }
  else if(lattice_shape_name == "ELLIPSOID") {
    for (int x=0; x<dim[0]; ++x) {
      for (int y=0; y<dim[1]; ++y) {
        for (int z=0; z<dim[2]; ++z) {
          const double a = 0.5*dim[0];
          const double b = 0.5*dim[1];
          const double c = 0.5*dim[2];

          if( ((x-a)*(x-a)/(a*a) + (y-b)*(y-b)/(b*b) + (z-c)*(z-c)/(c*c)) < 1.0) {

            for (int n=0; n<num_atoms; ++n) {

              const int type_number = unit_cell_types(n);
              atom_type.push_back(type_number);
              type_count[type_number]++;
              latt(x, y, z, n) = counter++;
            } // n
          }
        } // z
      } // y
    } // x
  }
  else {
    jams_error("Unknown lattice shape function: %s\n", lattice_shape_name.c_str());
  }

  num_spins = counter;
  num_spins3 = 3*num_spins;

  output.write("  * Total atoms in lattice: %i\n", num_spins);
}

void Lattice::calculateAtomPos(const jblib::Array<int, 1> &unit_cell_types, const jblib::Array<double, 2> &unit_cell_positions, jblib::Array<int, 4> &latt, std::vector<int> &dim, const double unitcell[3][3], const int num_atoms) {
  using namespace globals;
  assert(num_spins > 0);

  atom_pos.resize(num_spins, 3);
  local_atom_pos.resize(num_spins, 3);


  double r[3], p[3];
  int q[3];
  int atom_counter = 0;
  for (int x = 0; x < dim[0]; ++x) {
    for (int y = 0; y < dim[1]; ++y) {
      for (int z = 0; z < dim[2]; ++z) {
        for (int n = 0; n < num_atoms; ++n) {
          if (latt(x, y, z, n) != -1) {
            q[0] = x; q[1] = y; q[2] = z;
            for (int i = 0; i < 3; ++i) {
              p[i] = q[i] + unit_cell_positions(n, i);
            }
            matmul(unitcell, p, r);
            for (int i = 0; i < 3; ++i) {
              local_atom_pos(atom_counter, i) = q[i] + p[i];
              atom_pos(atom_counter, i) = r[i]*lattice_parameter;
            }
            atom_counter++;
          }
        }  // n
      }  // z
    }  // y
  }  // x
  assert(atom_counter == num_spins);
}

///
/// @brief  Print lattice to file.
///
void printLattice(const jblib::Array<int, 1> &unit_cell_types, const jblib::Array<double, 2> &unit_cell_positions, jblib::Array<int, 4> &latt, std::vector<int> &dim, const double unitcell[3][3], std::vector<int> &atom_type, const int num_atoms) {
  using namespace globals;
  assert(num_spins > 0);

  std::ofstream structure_file;
  structure_file.open("structure.xyz");
  structure_file << num_spins << "\n";
  structure_file << "Generated by JAMS++\n";

  for(int i = 0; i<num_spins; ++i){
    structure_file << "Type" << atom_type[i] << "\t" << atom_pos(i, 0) <<"\t"<< atom_pos(i, 1) <<"\t"<< atom_pos(i, 2) <<"\n";
  }

  //double r[3], p[3];
  //int q[3];
  //for (int x=0; x<dim[0]; ++x) {
    //for (int y=0; y<dim[1]; ++y) {
      //for (int z=0; z<dim[2]; ++z) {
        //for (int n=0; n<num_atoms; ++n) {
          //if(latt(x, y, z, n) != -1) {
            //structure_file << unit_cell_types(n) <<"\t";
            //q[0] = x; q[1] = y; q[2] = z;
            //for(int i = 0; i<3; ++i) {
              //r[i] = 0.0;
              //p[i] = unit_cell_positions(n, i);
              //for(int j = 0; j<3; ++j) {
                //r[i] += unitcell[j][i]*(q[j]+p[i]);
              //}
            //}
            //structure_file << r[0] <<"\t"<< r[1] <<"\t"<< r[2] <<"\n";
          //}
        //} // n
      //} // z
    //} // y
  //} // x
  structure_file.close();

}

///
/// @brief  Resize global arrays.
///
void resize_global_arrays() {
  using namespace globals;
  assert(num_spins > 0);
  s.resize(num_spins, 3);
  h.resize(num_spins, 3);
  alpha.resize(num_spins);
  mus.resize(num_spins);
  gyro.resize(num_spins);
}

///
/// @brief  initialize global arrays with material parameters.
///
void initialize_global_arrays(libconfig::Config &config, const libconfig::Setting &cfgMaterials, std::vector<int> &atom_type) {
  using namespace globals;

  output.write("\nInitialising global variables...\n");
  for(int i = 0; i<num_spins; ++i) {
    int type_num = atom_type[i];
    libconfig::Setting& type_settings = cfgMaterials[type_num];

    bool randomize_spins_is_set = false;
    type_settings.lookupValue("spinRand",randomize_spins_is_set);

    if(randomize_spins_is_set){
      rng.sphere(s(i, 0), s(i, 1), s(i, 2));

      for(int j = 0;j<3;++j){
        h(i, j) = 0.0;
      }
    }else{
      for(int j = 0;j<3;++j) {
        s(i, j) = type_settings["spin"][j];
      }
      double norm = sqrt(s(i, 0)*s(i, 0) + s(i, 1)*s(i, 1) + s(i, 2)*s(i, 2));

      for(int j = 0;j<3;++j){
        s(i, j) = s(i, j)/norm;
        h(i, j) = 0.0;
      }
    }

    mus(i) = type_settings["moment"];
      mus(i) = mus(i);  //*mu_bohr_si;

      alpha(i) = type_settings["alpha"];

      gyro(i) = type_settings["gyro"];
      gyro(i) = -gyro(i)/((1.0+alpha(i)*alpha(i))*mus(i));
    }
  }

///
/// @brief  Read the fourspin interaction parameters from configuration file.
///
  void readJ4Interactions(std::string &J4FileName, libconfig::Config &config, const jblib::Array<int, 1> &unit_cell_types, const jblib::Array<double, 2> &unit_cell_positions, jblib::Array<double, 4> &J4Vectors,
    jblib::Array<int, 3> &J4Neighbour, jblib::Array<double, 2> &J4Values, std::vector<int> &nJ4InteractionsOfType, const int num_atoms, std::map<std::string, int> &atom_type_map, const double unitcellInv[3][3], int &nJ4Values) {
    using namespace globals;

    output.write("\nReading fourspin interaction file...\n");

    double Jval;
    int nInterConfig = 0;
    double r[3];
    std::vector<int> atomInterCount(num_spins, 0);

    std::ifstream exchangeFile(J4FileName.c_str());

  // count number of interactions
    if( exchangeFile.is_open() ) {
      int nInterTotal=0;
      int atom1=0;
      for( std::string line; getline(exchangeFile, line); ) {
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
    for(int i = 0; i<num_spins; ++i) {
      if(atomInterCount[i] > interMax) {
        interMax = atomInterCount[i];
      }
    }

    output.write("  * Fourspin interactions in file: %d\n", nInterConfig);

  // Resize interaction arrays
    J4Vectors.resize(num_atoms, interMax, 3, 3);
    J4Neighbour.resize(num_atoms, interMax, 3);
    nJ4InteractionsOfType.resize(num_atoms, 0);

  //-----------------------------------------------------------------
  //  Read exchange tensor values from config
  //-----------------------------------------------------------------
    J4Values.resize(num_atoms, interMax);

  // zero jij array
    for(int i = 0; i<num_atoms; ++i) {
      for(int j = 0; j<interMax; ++j) {
        J4Values(i, j) = 0.0;
      }
    }

  // rewind file
    exchangeFile.clear();
    exchangeFile.seekg(0, std::ios::beg);

    int inter_counter = 0;
    for(int n=0; n<nInterConfig; ++n) {
      std::string line;

      getline(exchangeFile, line);
      std::stringstream is(line);

    // read exchange tensor

      double vij[3];

      int atom_num[4];

      for(int j = 0;j<4;++j){
        is >> atom_num[j];
      }

    // count from zero
      for(int j = 0;j<4;++j){
        atom_num[j]--;
      }

    // --------------- vij ----------------
      for (int j = 1; j < 4;++j) {
        for (int i = 0; i < 3; ++i) {
        is >> r[i];                               // real space vector to neighbour
      }
      matmul(unitcellInv, r, vij);                  // place interaction vector in unitcell space

      for(int i = 0; i<3; ++i) {
        J4Neighbour(atom_num[0], nJ4InteractionsOfType[atom_num[0]], j-1) = atom_num[j] - atom_num[0]; // store unitcell atom difference

        J4Vectors(atom_num[0], nJ4InteractionsOfType[atom_num[0]], j-1, i) = vij[i];
      }
    }

    is >> Jval; // bilinear
    J4Values(atom_num[0], nJ4InteractionsOfType[atom_num[0]]) = Jval/mu_bohr_si; // Jxx Jyy Jzz

    inter_counter++;
    nJ4InteractionsOfType[atom_num[0]]++;
  }



}
///
/// @brief  Read the interaction parameters from configuration file.
///
void read_interactions(std::string &exchangeFileName, libconfig::Config &config, const jblib::Array<int, 1> &unit_cell_types, const jblib::Array<double, 2> &unit_cell_positions, jblib::Array<double, 3> &interactionVectors,
  jblib::Array<int, 2> &interactionNeighbour, jblib::Array<double, 4> &JValues, jblib::Array<double, 2> &J2Values, std::vector<int> &nInteractionsOfType, const int num_atoms, std::map<std::string, int> &atom_type_map, const double unitcellInv[3][3], bool &J2Toggle, int &nJValues) {
  using namespace globals;

  output.write("\nReading interaction file...\n");

  if( !config.lookupValue("lattice.biquadratic", J2Toggle) ) {
    J2Toggle = false;
  }else if (!J2Toggle){
    output.write("  * Biquadratic exchange ON\n");
    output.write("\n************************************************************************\n");
    output.write("Biquadratic values will be read from the last column of the exchange file\n");
    output.write("************************************************************************\n\n");
  }

  int nInterTotal=0;

  int nInterConfig = 0;

  double r[3];

  std::vector<int> atomInterCount(num_spins, 0);

  std::ifstream exchangeFile(exchangeFileName.c_str());

  if( exchangeFile.is_open() ) {
    int atom1=0;
    for( std::string line; getline(exchangeFile, line); ) {
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
  for(int i = 0; i<num_spins; ++i) {
    if(atomInterCount[i] > interMax) {
      interMax = atomInterCount[i];
    }
  }

  output.write("  * Interactions in file: %d\n", nInterConfig);
  output.write("  * Total interactions (with symmetry): %d\n", nInterTotal);


  // Find number of exchange tensor components specified in the
  // config

  // rewind file
  exchangeFile.clear();
  exchangeFile.seekg(0, std::ios::beg);

  nJValues=0;
  if(nInterTotal > 0) {
    std::string line;
    int atom1=0;
    int atom2=0;

    getline(exchangeFile, line);
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

  if (verbose_output_is_set) {
    output.write("\nFound %d exchange components\n", nJValues);
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
  interactionVectors.resize(num_atoms, interMax, 3);
  interactionNeighbour.resize(num_atoms, interMax);
  nInteractionsOfType.resize(num_atoms, 0);

  // Resize global J1ij_t and J2ij_t matrices
  J1ij_s.resize(num_spins, num_spins);
  J1ij_t.resize(num_spins3, num_spins3);
  J2ij_t.resize(num_spins3, num_spins3);


  if( J2Toggle == true ){
    // Resize biquadratic matrix
    // NOTE: this matrix is NxN because we use a custom routine so the
    // matrix is just a convenient neighbour list.
    J2ij_s.resize(num_spins, num_spins);
    J2ij_s.setMatrixType(SPARSE_MATRIX_TYPE_SYMMETRIC);
    J2ij_s.setMatrixMode(SPARSE_MATRIX_MODE_LOWER);
  }

  // Set matrix types
  std::string solname;
  if( config.exists("sim.solver") == true ) {
    config.lookupValue("sim.solver", solname);
    solname = capitalize(solname);
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
  JValues.resize(num_atoms, interMax, 3, 3);
  J2Values.resize(num_atoms, interMax);

  // zero jij array
  for(int i = 0; i<num_atoms; ++i) {
    for(int j = 0; j<interMax; ++j) {
      J2Values(i, j) = 0.0;
      for(int k=0; k<3; ++k) {
        for(int l=0; l<3; ++l) {
          JValues(i, j, k, l) = 0.0;
        }
      }
    }
  }


  // rewind file
  exchangeFile.clear();
  exchangeFile.seekg(0, std::ios::beg);

  int inter_counter = 0;
  for(int n=0; n<nInterConfig; ++n) {
    std::string line;

    getline(exchangeFile, line);
    std::stringstream is(line);

    double d_latt[3]={0.0, 0.0, 0.0};
    // read exchange tensor

    int atom_num_1=0;
    int atom_num_2=0;

    is >> atom_num_1;
    is >> atom_num_2;

    // count from zero
    atom_num_1--; atom_num_2--;

    for(int i = 0; i<3; ++i) {
      is >> r[i];
    }

      // place interaction vector in unitcell space
    matmul(unitcellInv, r, d_latt);

      // store unitcell atom difference
    interactionNeighbour(atom_num_1, nInteractionsOfType[atom_num_1]) = atom_num_2 - atom_num_1;

      // store interaction vectors
    for(int i = 0;i<3; ++i){
      interactionVectors(atom_num_1, nInteractionsOfType[atom_num_1], i) = d_latt[i];
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
          for(int i = 0; i<3; ++i){
            JValues(atom_num_1, nInteractionsOfType[atom_num_1], i, i) = Jval/mu_bohr_si; // Jxx Jyy Jzz
          }
          break;
          case 2:
          is >> Jval;
          for(int i = 0; i<2; ++i){
            JValues(atom_num_1, nInteractionsOfType[atom_num_1], i, i) = Jval/mu_bohr_si; // Jxx Jyy
          }
          is >> Jval;
          JValues(atom_num_1, nInteractionsOfType[atom_num_1], 2, 2) = Jval/mu_bohr_si; // Jzz
          break;
          case 3:
          for(int i = 0; i<3; ++i){
            is >> Jval;
            JValues(atom_num_1, nInteractionsOfType[atom_num_1], i, i) = Jval/mu_bohr_si; // Jxx Jyy Jzz
          }
          break;
          case 9:
          for(int i = 0; i<3; ++i) {
            for(int j = 0; j<3; ++j) {
              is >> Jval;
              JValues(atom_num_1, nInteractionsOfType[atom_num_1], i, j) = Jval/mu_bohr_si;
            }
          }
          break;
          default:
          jams_error("Undefined exchange symmetry. 1, 2, 3 or 9 components must be specified\n");
        }

      // extra column at the end if biquadratic is on
        if(J2Toggle == true){
          is >> Jval;
          J2Values(atom_num_1, nInteractionsOfType[atom_num_1]) = Jval/mu_bohr_si;
        }

        inter_counter++;
        nInteractionsOfType[atom_num_1]++;

      }
    }



///
/// @brief  Create J4 interaction matrix.
///
    void createJ4Matrix(libconfig::Config &config, const libconfig::Setting &cfgMaterials, jblib::Array<int, 4> &latt,
      const std::vector<int> &dim, const int num_atoms, const jblib::Array<int, 1> &unit_cell_types, const jblib::Array<double, 2> &unit_cell_positions, const std::vector<int> &atom_type, const jblib::Array<double, 4> &J4Vectors,
      const jblib::Array<int, 3> &J4Neighbour, const jblib::Array<double, 2> &J4Values, const std::vector<int> &nJ4InteractionsOfType,
      const double unitcellInv[3][3], const bool pbc[3], const int &nJ4Values)
    {

      using namespace globals;
      output.write("\nCalculating fourspin interaction matrix...\n");

  const double encut = 1E-28/mu_bohr_si; // energy cutoff

  int qi[3], qj[3];
  double pj[3];
  int vij[3];
  int sj[3];

  J4ijkl_s.resize(num_spins, num_spins, num_spins, num_spins);

  int counter = 0;
  for (int x=0; x<dim[0]; ++x) {
    for (int y=0; y<dim[1]; ++y) {
      for (int z=0; z<dim[2]; ++z) {
        for (int n=0; n<num_atoms; ++n) {

          if(latt(x, y, z, n) != -1) {

            const int si = latt(x, y, z, n); assert (si < num_spins);

            qi[0] = x; qi[1] = y; qi[2] = z;

            for(int i = 0; i<nJ4InteractionsOfType[n]; ++i) {

              // loop the 3 neighbours of the fourspin term
              for(int snbr=0; snbr<3; ++snbr){

                int m = (J4Neighbour(n, i, snbr)+n)%num_atoms; assert(m >=0);

                for(int j = 0; j<3; ++j){
                  pj[j] = unit_cell_positions(m, j);
                }

                // put pj in unitcell
                double pj_cell[3] = {0.0, 0.0, 0.0};
                matmul(unitcellInv, pj, pj_cell);

                for(int j = 0; j<3; ++j){
                  qj[j] = floor(J4Vectors(n, i, snbr, j) - pj_cell[j] + 0.5);
                }

                // calculate realspace interaction vector
                bool idxcheck = true;
                for(int j = 0; j<3; ++j){
                  vij[j] = qi[j] - qj[j];
                  if(pbc[j] == true) {
                    vij[j] = (dim[j]+vij[j])%dim[j];
                  } else if (vij[j] < 0 || !(vij[j] < dim[j])) {
                    idxcheck = false;
                  }
                }

                // check interaction existed
                if(idxcheck == true) {
                  sj[snbr] = latt(vij[0], vij[1], vij[2], m); assert(sj[snbr] < num_spins);
                } else {
                  sj[snbr] = -1;
                }
              }


              if( (sj[0] > -1) && (sj[1] > -1) && (sj[2] > -1) ){
                if( fabs(J4Values(n, i)) > encut) {
                  J4ijkl_s.insert(si, sj[0], sj[1], sj[2], J4Values(n, i));
                }
                counter++;
              }


            }
          }
        }
      }
    }
  }

  J4ijkl_s.finalize();

  output.write("  * Total J4 interaction count: %i\n", counter);
  output.write("  * J4ijkl_s matrix memory (MAP): %f MB\n", J4ijkl_s.calculateMemoryUsage());
}


///
/// @brief  Create interaction matrix.
///
void createInteractionMatrix(libconfig::Config &config, const libconfig::Setting &cfgMaterials, jblib::Array<int, 4> &latt,
  const std::vector<int> &dim, const int num_atoms, const jblib::Array<int, 1> &unit_cell_types, const jblib::Array<double, 2> &unit_cell_positions, const std::vector<int> &atom_type, const jblib::Array<double, 3> &interactionVectors,
  const jblib::Array<int, 2> &interactionNeighbour, const jblib::Array<double, 4> &JValues, const std::vector<int> &nInteractionsOfType,
  const double unitcellInv[3][3], const bool pbc[3], const bool &J2Toggle, const jblib::Array<double, 2> &J2Values, const int &nJValues)
{

  using namespace globals;
  output.write("\nCalculating interaction matrix...\n");

  const double encut = 1E-26/mu_bohr_si; // energy cutoff

  double pnbr[3];
  int v[3], q[3], qnbr[3];

  int counter = 0;
  for (int x=0; x<dim[0]; ++x) {
    for (int y=0; y<dim[1]; ++y) {
      for (int z=0; z<dim[2]; ++z) {
        for (int n=0; n<num_atoms; ++n) {

          if(latt(x, y, z, n) != -1) {

            const int s_i = latt(x, y, z, n);
            const int type_num = atom_type[s_i];

            assert(s_i < num_spins);

            q[0] = x; q[1] = y; q[2] = z;

            int localInteractionCount = 0;
            for(int i = 0; i<nInteractionsOfType[n]; ++i) {
              // neighbour atom number
              int m = (interactionNeighbour(n, i)+n)%num_atoms;

              assert(m >= 0);

              for(int j = 0; j<3; ++j) { pnbr[j] =  unit_cell_positions(m, j); }

                double pnbrcell[3]={0.0, 0.0, 0.0};
              // put pnbr in unit cell
              matmul(unitcellInv, pnbr, pnbrcell);

              for(int j = 0; j<3; ++j) { qnbr[j] = floor(interactionVectors(n, i, j)-pnbrcell[j]+0.5); }

                bool idxcheck = true;
              for(int j = 0; j<3; ++j) {
                v[j] = q[j]+qnbr[j];
                if(pbc[j] == true) {
                  v[j] = (dim[j]+v[j])%dim[j];
                } else if (v[j] < 0 || !(v[j] < dim[j])) {
                  idxcheck = false;
                }
              }

              if(idxcheck == true && (latt(v[0], v[1], v[2], m) != -1) ) {
                int nbr = latt(v[0], v[1], v[2], m);
                assert(nbr < num_spins);
                //assert(s_i != nbr); self interaction is ok for example
                //FeRh

//---------------------------------------------------------------------
// Bilinear interactions
//---------------------------------------------------------------------
                if(nJValues == 1){
                //--------//
                // Scalar //
                //--------//
                  if(fabs(JValues(n, i, 0, 0)) > encut) {
                    if(J1ij_s.getMatrixType() == SPARSE_MATRIX_TYPE_SYMMETRIC) {
                      if(J1ij_s.getMatrixMode() == SPARSE_MATRIX_MODE_LOWER) {
                        if(s_i >= nbr){
                          J1ij_s.insertValue(s_i, nbr, JValues(n, i, 0, 0));
                        }
                      }else{
                        if(s_i <= nbr){
                          J1ij_s.insertValue(s_i, nbr, JValues(n, i, 0, 0));
                        }
                      }
                    }else{
                      J1ij_s.insertValue(s_i, nbr, JValues(n, i, 0, 0));
                    }
                  }
                } else {
                //--------//
                // Tensor //
                //--------//
                  for(int row=0; row<3; ++row) {
                    for(int col=0; col<3; ++col) {
                      if(fabs(JValues(n, i, row, col)) > encut) {
                        if(J1ij_t.getMatrixType() == SPARSE_MATRIX_TYPE_SYMMETRIC) {
                          if(J1ij_t.getMatrixMode() == SPARSE_MATRIX_MODE_LOWER) {
                            if(s_i > nbr){
                              J1ij_t.insertValue(3*s_i+row, 3*nbr+col, JValues(n, i, row, col));
                            }
                          } else {
                            if(s_i < nbr){
                              J1ij_t.insertValue(3*s_i+row, 3*nbr+col, JValues(n, i, row, col));
                            }
                          }
                        } else {
                          J1ij_t.insertValue(3*s_i+row, 3*nbr+col, JValues(n, i, row, col));
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
                  if(fabs(J2Values(n, i)) > encut) {
                    if(J2ij_s.getMatrixType() == SPARSE_MATRIX_TYPE_SYMMETRIC) {
                      if(J2ij_s.getMatrixMode() == SPARSE_MATRIX_MODE_LOWER) {
                        if(s_i >= nbr){
                          J2ij_s.insertValue(s_i, nbr, J2Values(n, i));
                        }
                      } else {
                        if(s_i <= nbr){
                          J2ij_s.insertValue(s_i, nbr, J2Values(n, i));
                        }
                      }
                    } else {
                      J2ij_s.insertValue(s_i, nbr, J2Values(n, i));
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
            libconfig::Setting& type_settings = cfgMaterials[type_num];

            double anival = type_settings["uniaxialAnisotropy"][1];
            double e3[3] = {0.0, 0.0, 0.0};

            // NOTE: Technically anisotropic is biquadratic but
            // biquadratic interactions in JAMS++ are implemented as a
            // scalar not a tensor so we store these in a seperate
            // array
            bool random_anisotropy_is_set = false;
            type_settings.lookupValue("random_anisotropy", random_anisotropy_is_set);

            // read anisotropy axis unit vector and ensure it is normalised
            if (random_anisotropy_is_set) {
              rng.sphere(e3[0], e3[1], e3[2]);
            } else {
              for (int i = 0; i < 3; ++i) {
                e3[i] = type_settings["uniaxialAnisotropy"][0][i];
              }
              double norm = 1.0/sqrt(e3[0]*e3[0]+e3[1]*e3[1]+e3[2]*e3[2]);

              for(int i = 0;i<3;++i) {
                e3[i] = e3[i]*norm;
              }
            }

            const double DTensor[3][3] = { { e3[0]*e3[0], e3[0]*e3[1], e3[0]*e3[2] },
            { e3[1]*e3[0], e3[1]*e3[1], e3[1]*e3[2] },
            { e3[2]*e3[0], e3[2]*e3[1], e3[2]*e3[2] } };

            if (verbose_output_is_set) {
              output.write("\nAnisotropy Tensor\n");
              output.write("%f, %f, %f\n", DTensor[0][0], DTensor[0][1], DTensor[0][2]);
              output.write("%f, %f, %f\n", DTensor[1][0], DTensor[1][1], DTensor[1][2]);
              output.write("%f, %f, %f\n", DTensor[2][0], DTensor[2][1], DTensor[2][2]);
            }

            // NOTE: Factor of 2 is accounted for in the biquadratic
            // calculation
            const double Dval = anival/mu_bohr_si;


            for (int row = 0; row < 3; ++row) {
              for (int col = 0; col < 3; ++col) {
                if((Dval*DTensor[row][col]) > encut) {
                  if(J2ij_t.getMatrixType() == SPARSE_MATRIX_TYPE_SYMMETRIC) {
                    if(J2ij_t.getMatrixMode() == SPARSE_MATRIX_MODE_LOWER) {
                      if(row >= col){
                        J2ij_t.insertValue(3*s_i+row, 3*s_i+col, Dval*DTensor[row][col]);
                      }
                    } else {
                      if(row <= col){
                        J2ij_t.insertValue(3*s_i+row, 3*s_i+col, Dval*DTensor[row][col]);
                      }
                    }
                  } else {
                    J2ij_t.insertValue(3*s_i+row, 3*s_i+col, Dval*DTensor[row][col]);
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
  output.write("  * J1ij_t matrix memory (MAP): %f MB\n", J1ij_t.calculateMemory());

// J1ij_t.printCSR();
}



void Lattice::create_from_config(libconfig::Config &config) {
  using namespace globals;

  output.write("\nCalculating lattice...\n");

  try {

    jblib::Array<int, 4> latt;
    jblib::Array<double, 2> unit_cell_positions;
    jblib::Array<int, 1>      unit_cell_types;
    jblib::Array<double, 3> interactionVectors;
    jblib::Array<double, 4> JValues;
    jblib::Array<double, 2> J2Values;
    jblib::Array<int, 2> interactionNeighbour;
    jblib::Array<int, 3> J4Neighbour;
    jblib::Array<double, 4> J4Vectors;
    jblib::Array<double, 2> J4Values;
    std::vector<int> nInteractionsOfType;
    std::vector<int> nJ4InteractionsOfType;
    jblib::Array<double, 3> jij;
    int num_atoms=0;
    bool pbc[3] = {true, true, true};
    bool J2Toggle = false;
    int nJValues=0;

    double unitcell[3][3];
    double unitcellInv[3][3];


    const libconfig::Setting& cfgLattice    =   config.lookup("lattice");
    const libconfig::Setting& cfgBasis      =   config.lookup("lattice.basis");
    const libconfig::Setting& cfgMaterials  =   config.lookup("materials");
//    const libconfig::Setting& cfgExchange   =   config.lookup("exchange");


    read_basis(cfgBasis, unitcell, unitcellInv);

    std::string atom_position_filename = config.lookup("lattice.positions");
    readAtoms(atom_position_filename, unit_cell_types, unit_cell_positions, num_atoms, num_atom_types, atom_type_map, atom_names);

    assert(num_atoms > 0);
    assert(num_atom_types > 0);

    readLattice(cfgLattice, dim, pbc, unitcell);

    type_count.resize(num_atom_types);
    for(int i = 0; i<num_atom_types; ++i) { type_count[i] = 0; }

      create_lattice(cfgLattice, unit_cell_types, unit_cell_positions, atom_type_map, latt, atom_type, type_count, dim, num_atoms, pbc);

    calculateAtomPos(unit_cell_types, unit_cell_positions, latt, dim, unitcell, num_atoms);
#ifdef DEBUG
    printLattice(unit_cell_types, unit_cell_positions, latt, dim, unitcell, atom_type, num_atoms);
#endif // DEBUG

    resize_global_arrays();

    initialize_global_arrays(config, cfgMaterials, atom_type);

    std::string exchangeFileName = config.lookup("lattice.exchange");
    read_interactions(exchangeFileName, config, unit_cell_types, unit_cell_positions, interactionVectors, interactionNeighbour, JValues, J2Values, nInteractionsOfType,
      num_atoms, atom_type_map, unitcellInv, J2Toggle, nJValues);

    createInteractionMatrix(config, cfgMaterials, latt, dim, num_atoms, unit_cell_types, unit_cell_positions,
      atom_type, interactionVectors, interactionNeighbour, JValues,
      nInteractionsOfType, unitcellInv, pbc, J2Toggle, J2Values, nJValues);

    map_position_to_int();

    if( config.exists("lattice.fourspin") == true ) {
      int nJ4Values=0;
      std::string J4FileName = config.lookup("lattice.fourspin");

      readJ4Interactions(J4FileName, config, unit_cell_types, unit_cell_positions, J4Vectors, J4Neighbour, J4Values, nJ4InteractionsOfType,
        num_atoms, atom_type_map, unitcellInv, nJ4Values);

      createJ4Matrix(config, cfgMaterials, latt, dim, num_atoms, unit_cell_types, unit_cell_positions,
        atom_type, J4Vectors, J4Neighbour, J4Values,
        nJ4InteractionsOfType, unitcellInv, pbc, nJ4Values);
    }

    if (config.exists("lattice.coarse") == true ) {
      initialize_coarse_magnetisation_map();
    }

  } // try
  catch(const libconfig::SettingNotFoundException &nfex) {
    jams_error("Setting '%s' not found", nfex.getPath());
  }

}

void Lattice::output_spin_state_as_vtu(std::ofstream &outfile){
  using namespace globals;

  outfile << "<?xml version=\"1.0\"?>" << "\n";
  outfile << "<VTKFile type=\"UnstructuredGrid\">" << "\n";
  outfile << "<UnstructuredGrid>" << "\n";
  outfile << "<Piece NumberOfPoints=\""<<num_spins<<"\"  NumberOfCells=\"1\">" << "\n";
  outfile << "<PointData Scalar=\"Spins\">" << "\n";

  for(int n=0; n<num_atom_types; ++n){
    outfile << "<DataArray type=\"Float32\" Name=\"" << atom_names[n] << "Spin\" NumberOfComponents=\"3\" format=\"ascii\">" << "\n";
    for(int i = 0; i<num_spins; ++i){
      if(atom_type[i] == n){
        outfile << s(i, 0) << "\t" << s(i, 1) << "\t" << s(i, 2) << "\n";
      } else {
        outfile << 0.0 << "\t" << 0.0 << "\t" << 0.0 << "\n";
      }
    }
    outfile << "</DataArray>" << "\n";
  }
  outfile << "</PointData>" << "\n";
  outfile << "<CellData>" << "\n";
  outfile << "</CellData>" << "\n";
  outfile << "<Points>" << "\n";
  outfile << "<DataArray type=\"Float32\" NumberOfComponents=\"3\" format=\"ascii\">" << "\n";
  for(int i = 0; i<num_spins; ++i){
    outfile << atom_pos(i, 0) << "\t" << atom_pos(i, 1) << "\t" << atom_pos(i, 2) << "\n";
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


void Lattice::map_position_to_int(){
  using namespace globals;

  spin_int_map.resize(num_spins, 3);

  for(int i = 0; i<num_spins; ++i){
    for(int j = 0; j<3; ++j){
      spin_int_map(i, j) = nint((local_atom_pos(i, j))*unitcell_kpoints[j]);
    }
        //std::cout<<atom_pos(i, 0)<<"\t"<<atom_pos(i, 1)<<"\t"<<atom_pos(i, 2)<<"\t";
        //std::cout<<spin_int_map(i, 0)<<"\t"<<spin_int_map(i, 1)<<"\t"<<spin_int_map(i, 2)<<std::endl;
  }
}


void Lattice::output_spin_state_as_binary(std::ofstream &outfile){
  using namespace globals;

  outfile.write(reinterpret_cast<char*>(&num_spins), sizeof(int));
  outfile.write(reinterpret_cast<char*>(s.data()), num_spins3*sizeof(double));
}

void Lattice::output_spin_types_as_binary(std::ofstream &outfile){
  using namespace globals;

  outfile.write(reinterpret_cast<char*>(&num_spins), sizeof(int));
  outfile.write(reinterpret_cast<char*>(&atom_type[0]), num_spins*sizeof(int));
}

void Lattice::read_spin_state_from_binary(std::ifstream &infile){
  using namespace globals;

  infile.seekg(0);

  int filenum_spins=0;
  infile.read(reinterpret_cast<char*>(&filenum_spins), sizeof(int));

  if(filenum_spins != num_spins){
    jams_error("I/O error, spin state file has %d spins but simulation has %d spins", filenum_spins, num_spins);
  }else{
    infile.read(reinterpret_cast<char*>(s.data()), num_spins3*sizeof(double));
  }

  if(infile.bad()){
    jams_error("I/O error. Unknown failure reading spin state file");
  }
}

void Lattice::initialize_coarse_magnetisation_map() {
  using namespace globals;

  const libconfig::Setting& cfgLattice    =   config.lookup("lattice");

  int register j, n;
  float resolution[3];

  coarseDim.resize(3);

  for (j = 0; j!=3; ++j) {
    coarseDim[j] = cfgLattice["coarse"][j];
    resolution[j] = coarseDim[j]/rmax[j];
  }

  spin_to_coarse_cell_map.resize(num_spins, 3);
  coarse_magnetisation_type_count.resize(coarseDim[0], coarseDim[1], coarseDim[2], num_atom_types);
  coarseMagnetisation.resize(coarseDim[0], coarseDim[1], coarseDim[2], num_atom_types, 3);

  // atom_pos is realspace atom position in nm
  for (n=0; n!=num_spins; ++n) {
    for (j = 0; j!=3; ++j) {
      spin_to_coarse_cell_map(n, j) = int(atom_pos(n, j)*resolution[j]);
    }
    coarse_magnetisation_type_count(spin_to_coarse_cell_map(n, 0), spin_to_coarse_cell_map(n, 1), spin_to_coarse_cell_map(n, 2), atom_type[n])++;
  }

}

void Lattice::output_coarse_magnetisation(std::ofstream &outfile) {
  using namespace globals;

  register int n, j, x, y, z, type;
  for (x=0; x!=coarseDim[0]; ++x) {
    for (y=0; y!=coarseDim[1]; ++y) {
      for (z=0; z!=coarseDim[2]; ++z) {
        for (type = 0; type!=num_atom_types; ++type) {
          for (j = 0; j!=3; ++j) {
            coarseMagnetisation(x, y, z, type, j) = 0.0;
          }
        }
      }
    }
  }
  // bin all the spins
  for (n=0; n!=num_spins; ++n) {

    x = spin_to_coarse_cell_map(n, 0);
    y = spin_to_coarse_cell_map(n, 1);
    z = spin_to_coarse_cell_map(n, 2);

    for (j = 0; j!=3; ++j) {
      coarseMagnetisation(x, y, z, atom_type[n], j) += s(n, j);
    }
  }

  // normalise the result
  for (x=0; x!=coarseDim[0]; ++x) {
    for (y=0; y!=coarseDim[1]; ++y) {
      for (z=0; z!=coarseDim[2]; ++z) {
        outfile << x <<"\t" << y << "\t" << z;
        for (type = 0; type!=num_atom_types; ++type) {
          for (j = 0; j!=3; ++j) {
            outfile << "\t" << (coarseMagnetisation(x, y, z, type, j) / static_cast<double>(coarse_magnetisation_type_count(x, y, z, type) ) );
          }
        }
        outfile << "\n";
      }
    }
  }
}
