// Copyright 2014 Joseph Barker. All rights reserved.

#include "physics/spinwaves.h"

#include <algorithm>
#include <cmath>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <map>
#include <string>

#include "core/globals.h"
#include "core/maths.h"

#include "jblib/containers/array.h"

SpinwavesPhysics::SpinwavesPhysics(const libconfig::Setting &settings)
: Physics(settings),
qDim(3, 0),
qSpace(),
qSpaceFFT(0),
componentReal(0),
componentImag(0),
coFactors(0, 0),
spinToKspaceMap(0),
nBZPoints(0),
BZIndex(0),
BZPoints(0, 0),
BZDegeneracy(0),
BZLengths(0),
BZData(0),
SPWFile(),
ModeFile(),
SPDFile(),
typeOverride(),
initialized(false),
squareToggle(false),
pumpTime(0.0),
pumpStartTime(0.0),
pumpTemp(0.0),
pumpFluence(0.0),
electronTemp(0.0),
phononTemp(0.0),
reversingField(3, 0.0),
Ce(7.0E02),
Cl(3.0E06),
G(17.0E17),
TTMFile() {
  using namespace globals;

  register int i, j, n;  // fast loop variables

  output.write("  * Spinwaves physics module\n");

  squareToggle = settings["SquarePulse"];
  if (squareToggle == true) {
    pumpTemp = settings["PumpTemp"];
  } else {
    pumpTemp = settings["InitialTemperature"];
  }

  phononTemp = settings["InitialTemperature"];
  electronTemp = phononTemp;

  // unitless according to Tom's code!
  pumpFluence = settings["PumpFluence"];
  pumpFluence = pumpPower(pumpFluence);

  // width of gaussian heat pulse in seconds
  pumpTime = settings["PumpTime"];

  pumpStartTime = settings["PumpStartTime"];

  for (i = 0; i < 3; ++i) {
    reversingField[i] = settings["ReversingField"][i];
  }



  std::string fileName = "_ttm.dat";
  fileName = seedname+fileName;
  TTMFile.open(fileName.c_str());

  TTMFile << std::setprecision(8);

  TTMFile << "# t [s]\tT_el [K]\tT_ph [K]\tLaser [arb/]\n";

  //-------------------------------------------------------------------
{   // Choose components to transform.
  // x = 0 ; y = 1 ; z = 2
  // If two components are specified the the Fourier transform will be
    // performed on (for example) x + iy.
    //
    // If only one component is specified then the Fourier transform is
    // performed on (fore example) z + i0. The imaginary component is set
    // to -1 which can be detected later to allow different calculations
    // for single component transforms.
}   //
    //-------------------------------------------------------------------

std::string strImag, strReal;
std::map<std::string, int> componentMap;

componentMap["X"] = 0; componentMap["Y"] = 1; componentMap["Z"] = 2;

config.lookupValue("physics.componentReal", strReal);
std::transform(strReal.begin(), strReal.end(), strReal.begin(), toupper);

componentReal = componentMap[strReal];

if(config.exists("physics.componentImag")) {
  config.lookupValue("physics.componentImag", strImag);

  std::transform(strImag.begin(), strImag.end(), strImag.begin(), toupper);

  componentImag = componentMap[strImag];
  output.write("  * Fourier transform component: (%s, i%s)\n", strReal.c_str(),
    strImag.c_str());
} else {
    componentImag = -1;  // dont use imaginary component
    output.write("  * Fourier transform component: %s\n", strReal.c_str());
  }

  //-------------------------------------------------------------------
{  // Read type cofactors
  //
  // Each material must define a vector (3 element list) called "coFactors".
  // These values are then used to multiply the spin elements before the
  // FFT is performed. This may be used for example to do a Holstein-
  // Primakoff transformation in an anti-ferromagnet. So one sublattice would
  // have cofactors [1, 1, 1] and the other sublattice would have [1, -1, -1]
  // which performs the operation Sx -> Sx ; Sy -> -Sy ; Sz -> -Sz.
}  //
  //-------------------------------------------------------------------

libconfig::Setting &mat = config.lookup("materials");
coFactors.resize(lattice.numTypes(), 3);
for (i = 0; i < lattice.numTypes(); ++i) {
  for (j = 0; j < 3; ++j) {
    coFactors(i, j) = mat[i]["coFactors"][j];
  }
}

  //-------------------------------------------------------------------
{  // Read k-space dimensions
  //
  // Note: this returns the number of k-points in the unit cell
  // multiplied by the number of unit cells in each dimension.
}  //
  //-------------------------------------------------------------------

lattice.getKspaceDimensions(qDim[0], qDim[1], qDim[2]);
output.write("  * Kspace Size [%d, %d, %d]\n", qDim[0], qDim[1], qDim[2]);

  //-------------------------------------------------------------------
{  // Read irreducible Brillouin zone (IBZ)
  //
  // In the configuration file the user can set a series symmetry points
  // (or in fact any points) between which they would like the DSF.
  // Usually this would be the IBZ or some part of it. We must calculate
  // all of the intermediate points between each specified symmetry point
  // and also we exploit the symmetry of the reciprocal space. This
  // allows us to have a better averaging in k-space and also to
  // calculate the density of states (DOS).
  //
  // TODO(joe): Implement a flag in the configuration to turn the symmetry
  // operations on and off.
}  //
  //--------------------------------------------------------------------

const int nSymPoints = settings["brillouinzone"].getLength();

jblib::Array<int, 2> SymPoints(nSymPoints, 3);
jblib::Array<int, 1> BZPointCount(nSymPoints-1);

for (i = 0; i < nSymPoints; ++i) {
  for (j = 0; j < 3; ++j) {
    SymPoints(i, j) = settings["brillouinzone"][i][j];
  }
}

  //-------------------------------------------------------------------
{  // Count BZ vector points
  //
  // This is the number of points in k-space between the symmetry points.
}   //
  //-------------------------------------------------------------------

nBZPoints = 0;
for (i = 0; i < (nSymPoints-1); ++i) {
  int max = 0;
  for (j = 0; j < 3; ++j) {
    int x = abs(SymPoints(i+1, j) - SymPoints(i, j));
    if (x > max) {
     max = x;
   }
 }
 BZPointCount(i) = max;
 nBZPoints += max;
}
nBZPoints += 1;  // include last symmetry point

  //-------------------------------------------------------------------
{  // Count total BZ points after symmetry
  //
  // We need to know in advance how many BZ points there are when we
  // apply the symmetry operations within k-space. We look through
  // each pair of symmetry vectors in order and calculate the
  // intermediate points. Then for each point we calculate all
  // symmetric points in k-space and count these.
  //
  // Notes: When calculating the symmetric points we use abs() so the
  // symmetry points in the config file are best defined all positive
  // anyway. This is done because next_point_symmetry() requires the
  // initial vector to be [+, +, +] and sorted to produce all
  // permutations.
}  //
  //-------------------------------------------------------------------

int vec[3] = {0, 0, 0};
register int counter = 0;

  // Loop through pairs of symmetry points
for (i = 0; i < (nSymPoints-1); ++i) {
  for (j = 0; j<3; ++j) {
    vec[j] = SymPoints(i+1, j)-SymPoints(i, j);
    if (vec[j] != 0) {
      vec[j] = vec[j] / abs(vec[j]);
    }
  }

  // Loop through intermediate points
  for (n = 0; n < BZPointCount(i); ++n) {
    int bzvec[3];

    for (j = 0; j < 3; ++j) { bzvec[j] = abs(SymPoints(i, j)+n*vec[j]); }

      // Calculate equivalent points in k-space
      std::sort(bzvec, bzvec+3);
    do {
      counter++;
    } while (next_point_symmetry(bzvec));
  }
}

  // Include the last symmetry point individually for nice output
{
  int bzvec[3];
  for (j = 0; j < 3; ++j) {
   bzvec[j] = abs(SymPoints(nSymPoints-1, j));
 }
 std::sort(bzvec, bzvec+3);
 do {
  counter++;
} while (next_point_symmetry(bzvec));
}

  //-------------------------------------------------------------------
{  // Store all BZ points of interest
  //
  // We now resize the arrays after the counting above and repeat the
  // procedure but store the results.
}  //
  //-------------------------------------------------------------------

  // Start and end indices for BZ Points
BZIndex.resize(nBZPoints+1);
  // All BZ points, +1 to add on final symmetry point
BZPoints.resize(counter+1, 3);
  // Length in k-space between points
BZLengths.resize(nBZPoints);
  // Degeneracy of the point under symmetry in k-space
BZDegeneracy.resize(nBZPoints);
BZData.resize(counter);

for (i = 0; i < nBZPoints; ++i) { BZDegeneracy(i)=0; }

    // counter of the points in the IBZ (i.e. don't count degenerate points)
  register int irreducibleCounter = 0;
counter = 0;

    // Loop through pairs of symmetry points
for (i = 0; i < (nSymPoints-1); ++i) {
  for (j = 0; j < 3; ++j) {
    vec[j] = SymPoints(i+1, j)-SymPoints(i, j);
    if (vec[j] != 0) {
      vec[j] = vec[j] / abs(vec[j]);
    }
  }

      // Loop through intermediate points
  for (n = 0; n < BZPointCount(i); ++n) {
    BZLengths(irreducibleCounter) =
    sqrt(vec[0]*vec[0]+vec[1]*vec[1]+vec[2]*vec[2]);
    BZIndex(irreducibleCounter) = counter;

    int bzvec[3];
    int pbcvec[3];

    for (j = 0; j < 3; ++j) {
      bzvec[j] = abs(SymPoints(i, j)+n*vec[j]);
    }

        // Calculate equivalent points in k-space
    std::sort(bzvec, bzvec+3);
    output.write("BZ Point: %8d %8d [ %4d %4d %4d ]\n", irreducibleCounter,
      counter, bzvec[0], bzvec[1], bzvec[2]);
    do {
          // Apply periodic boundaries
          //
          // Note: FFTW stores -q in reverse order at the end of the array.
          //
      for (int j = 0; j < 3; ++j) {
        pbcvec[j] = ((qDim[j])+bzvec[j])%(qDim[j]);
        BZPoints(counter, j) = pbcvec[j];
      }

      counter++;
      BZDegeneracy(irreducibleCounter)++;
    } while (next_point_symmetry(bzvec));
    irreducibleCounter++;
  }
  BZIndex(irreducibleCounter) = counter;
}

// Include the last symmetry point individually for nice output
{
  for (j = 0; j < 3; ++j) {
    vec[j] = SymPoints(nSymPoints-1, j);
    if (vec[j] != 0) {
      vec[j] = vec[j] / abs(vec[j]);
    }
  }
  BZLengths(irreducibleCounter)
  = sqrt(vec[0]*vec[0]+vec[1]*vec[1]+vec[2]*vec[2]);
  BZIndex(irreducibleCounter) = counter;

  int bzvec[3], pbcvec[3];

  for (j = 0; j < 3; ++j) {
    bzvec[j] = abs(SymPoints(nSymPoints-1, j));
  }
  std::sort(bzvec, bzvec+3);
  output.write("BZ Point: %8d %8d [ %4d %4d %4d ]\n", irreducibleCounter,
    counter, bzvec[0], bzvec[1], bzvec[2]);

  do {
    // Apply periodic boundaries
    //
    // Note: FFTW stores -q in reverse order at the end of the array.
    //

    for (j = 0; j < 3; ++j) {
      pbcvec[j] = ((qDim[j])+bzvec[j])%(qDim[j]);
      BZPoints(counter, j) = pbcvec[j];
    }
    BZDegeneracy(irreducibleCounter)++;
    counter++;
  } while (next_point_symmetry(bzvec));
  irreducibleCounter++;
}

BZIndex(irreducibleCounter) = counter;

  //-------------------------------------------------------------------
  //  Create map from spin number to col major index of qSpace array
  //-------------------------------------------------------------------


    // find min and max coordinates
int xmin, xmax;
int ymin, ymax;
int zmin, zmax;

    // populate initial values
lattice.getSpinIntCoord(0, xmin, ymin, zmin);
lattice.getSpinIntCoord(0, xmax, ymax, zmax);

for (int i = 0; i < num_spins; ++i) {
  int x, y, z;
  lattice.getSpinIntCoord(i, x, y, z);
  if (x < xmin) { xmin = x; }
  if (x > xmax) { xmax = x; }

  if (y < ymin) { ymin = y; }
  if (y > ymax) { ymax = y; }

  if (z < zmin) { zmin = z; }
  if (z > zmax) { zmax = z; }
}

output.write("  * qSpace range: [ %d:%d , %d:%d , %d:%d ]\n", xmin, xmax,
  ymin, ymax, zmin, zmax);

spinToKspaceMap.resize(num_spins);
for (int i = 0; i < num_spins; ++i) {
  int x, y, z;
  lattice.getSpinIntCoord(i, x, y, z);

  int idx = (x*qDim[1]+y)*qDim[2]+z;
  spinToKspaceMap[i] = idx;
}


  //-------------------------------------------------------------------
  // Allocate qSpace array
  //-------------------------------------------------------------------
output.write("  * qSpace allocating %f MB\n",
  (sizeof(fftw_complex)*qDim[0]*qDim[1]*qDim[2])/(1024.0*1024.0));
qSpace = static_cast<fftw_complex*>(fftw_malloc(
  sizeof(fftw_complex)*qDim[0]*qDim[1]*qDim[2]));

if (qSpace == NULL) {
  jams_error("Failed to allocate qSpace FFT array");
}

qSpaceFFT.push_back(fftw_plan_dft_3d(qDim[0], qDim[1], qDim[2], qSpace,
  qSpace, FFTW_FORWARD, FFTW_MEASURE));

std::string filename = "_spw.dat";
filename = seedname+filename;
SPWFile.open(filename.c_str());
SPWFile << "# t [s]\tk=0\tk!=0\tM_AF1_x\tM_AF1_y\tM_AF1_z\n";

initialized = true;
}

SpinwavesPhysics::~SpinwavesPhysics() {
  if (initialized == true) {
    for (int i = 0; i < qSpaceFFT.size(); ++i) {
      fftw_destroy_plan(qSpaceFFT[i]);
    }
  }
  SPWFile.close();
  TTMFile.close();
}

void SpinwavesPhysics::update(const int &iterations, const double &time, const double &dt) {
  using namespace globals;
  const double relativeTime = (time-pumpStartTime);

  if (squareToggle == true) {
    if ((relativeTime > 0.0) && (relativeTime < pumpTime)) {
      temperature_ = pumpTemp;
    } else {
      temperature_ = phononTemp;
    }
  } else {
    if (relativeTime > 0.0) {
      for (int i = 0; i < 3; ++i) {
        applied_field_[i] = reversingField[i];
      }
      if (relativeTime <= 10*pumpTime) {
        pumpTemp = pumpFluence*exp(-((relativeTime-3*pumpTime)/(pumpTime))
          *((relativeTime-3*pumpTime)/(pumpTime)));
      } else {
        pumpTemp = 0.0;
      }
      electronTemp = electronTemp
      + ((-G*(electronTemp-phononTemp)+pumpTemp)*dt)/(Ce*electronTemp);
      phononTemp   = phononTemp
      + ((G*(electronTemp-phononTemp))*dt)/(Cl);
    }
    temperature_ = electronTemp;
  }

  if (iterations%output_step_freq_ == 0) {

    register int i, q;

    TTMFile << time << "\t" << electronTemp << "\t" << phononTemp << "\t" << pumpTemp << "\n";

    const int qTotal = qDim[0]*qDim[1]*qDim[2];

// Apply cofactors to transform spin components
    if (componentImag == -1) {
      for (i = 0; i < num_spins; ++i) {
        const int type = lattice.getType(i);
        const int idx = spinToKspaceMap[i];
        qSpace[idx][0] = coFactors(type, componentReal)*s(i, componentReal);
        qSpace[idx][1] = 0.0;
      }
    } else {
      for (i = 0; i < num_spins; ++i) {
        const int type = lattice.getType(i);
        const int idx = spinToKspaceMap[i];
        qSpace[idx][0] = coFactors(type, componentReal)*s(i, componentReal);
        qSpace[idx][1] = coFactors(type, componentImag)*s(i, componentImag);
      }
    }

    fftw_execute(qSpaceFFT[0]);

  // Normalise FFT by Nx*Ny*Nz
    for (int i = 0; i < (qDim[0]*qDim[1]*qDim[2]); ++i) {
      qSpace[i][0] /= qTotal;
      qSpace[i][1] /= qTotal;
    }

    for (int n = 0; n < nBZPoints; ++n) {
      BZData(n) = 0.0;
    }

    for (int n = 0; n < nBZPoints; ++n) {
      for (q = BZIndex(n); q < BZIndex(n+1); ++q) {
        const int qVec[3] = {BZPoints(q, 0), BZPoints(q, 1), BZPoints(q, 2)};
        const int qIdx = qVec[2] + qDim[2]*(qVec[1] + qDim[1]*qVec[0]);

        BZData(n) = BZData(n) + qSpace[qIdx][0]*qSpace[qIdx][0]
        + qSpace[qIdx][1]*qSpace[qIdx][1];
      }
    }

    float lengthTotal = 0.0;
    for (int n = 0; n < nBZPoints; ++n) {
      const int q = BZIndex(n);
      SPWFile << time << "\t" << lengthTotal << "\t" << BZPoints(q, 0)
      << "\t" << BZPoints(q, 1) << "\t" << BZPoints(q, 2);
      SPWFile << "\t" << BZData(n) << "\t"
      << static_cast<double>(BZDegeneracy(n)) << "\n";
      lengthTotal += BZLengths(n);
    }
    SPWFile << std::endl;
  }
}

