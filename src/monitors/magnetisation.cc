#include "globals.h"
#include "magnetisation.h"
#include "lattice.h"

#include <cmath>

void MagnetisationMonitor::initialise() {
  using namespace globals;
  output.write("\nInitialising Magnetisation monitor...\n");

  std::string name = "_mag.dat";
  name = seedname+name;
  outfile.open(name.c_str());

  // mx my mz |m| m2zz (quadrupole)
  mag.resize(lattice.numTypes(),5);

  initialised = true;
}

void MagnetisationMonitor::run() {

}

void MagnetisationMonitor::initConvergence(ConvergenceType type, const double tol){
	convType = type;
	tolerance = tol;
}

bool MagnetisationMonitor::checkConvergence(){
	if(convType == convNone){
		return false;
	} else { 
		output.write("Convergence: mean %e \t stddev %e [tolerance %e]\n",rs.mean(),rs.stdDev(),tolerance);	
		if(rs.stdDev() < tolerance){
			return true;
		}
	}
	return false;
}
void MagnetisationMonitor::write(const double &time) {
  using namespace globals;
  assert(initialised);
  int i,j,type;
  
  for(i=0; i<lattice.numTypes(); ++i) {
    for(j=0; j<5; ++j) {
      mag(i,j) = 0.0;
    }
  }

  for(i=0; i<nspins; ++i) {
    type = lattice.getType(i);
    for(j=0;j<3;++j) {
      mag(type,j) += s(i,j);
    }
  }
  for(i=0; i<nspins; ++i) {
    type = lattice.getType(i);
    mag(type,4) += ( (s(i,2)*s(i,2))-(1.0/3.0) );
  }
  
  for(i=0; i<lattice.numTypes(); ++i) {
    mag(i,4) = mag(i,4)/static_cast<double>(lattice.getTypeCount(i));
  }

  for(i=0; i<lattice.numTypes(); ++i) {
    for(j=0; j<3; ++j) {
      mag(i,j) = mag(i,j)/static_cast<double>(lattice.getTypeCount(i));
    }
  }

  for(i=0; i<lattice.numTypes(); ++i) {
    mag(i,3) = sqrt(mag(i,0)*mag(i,0) + mag(i,1)*mag(i,1) + mag(i,2)*mag(i,2));
  }
  

  outfile << time;

  outfile << "\t" << globalTemperature;

  for(i=0; i<3; ++i) {
    outfile << "\t" << h_app[i];  
  }

  for(i=0; i<lattice.numTypes(); ++i) {
    outfile <<"\t"<< mag(i,0) <<"\t"<< mag(i,1) <<"\t"<< mag(i,2) <<"\t"<< mag(i,3) <<"\t" << mag(i,4);
  }
#ifdef NDEBUG
  outfile << "\n";
#else
  outfile << std::endl;
#endif

	switch(convType){
	case convMag:
		rs.push(mag(0,4));
		break;
	case convPhi:
		rs.push(acos(mag(0,3)/mag(0,4)));
		break;
	case convSinPhi:
		rs.push(sin(acos(mag(0,3)/mag(0,4))));
		break;
	default:
		break;
	}
	
}

MagnetisationMonitor::~MagnetisationMonitor() {
  outfile.close();
}
