#include "globals.h"
#include "consts.h"
#include "fields.h"

#include "metropolismc.h"

void MetropolisMCSolver::initialise(int argc, char **argv, double idt)
{
    using namespace globals;

    // initialise base class
    Solver::initialise(argc,argv,idt);

    output.write("Initialising Metropolis Monte-Carlo solver\n");

    output.write("  * Converting MAP to CSR\n");

    J1ij_s.convertMAP2CSR();
    J1ij_t.convertMAP2CSR();
    J2ij_s.convertMAP2CSR();
    J2ij_t.convertMAP2CSR();

    output.write("  * J1ij Scalar matrix memory (CSR): %f MB\n",J1ij_s.calculateMemory());
    output.write("  * J1ij Tensor matrix memory (CSR): %f MB\n",J1ij_t.calculateMemory());
    output.write("  * J2ij Scalar matrix memory (CSR): %f MB\n",J2ij_s.calculateMemory());
    output.write("  * J2ij Tensor matrix memory (CSR): %f MB\n",J2ij_t.calculateMemory());

    initialised = true;
}

void MetropolisMCSolver::run()
{

}

void MetropolisMCSolver::syncOutput(){
}

void MetropolisMCSolver::calcEnergy(double &e1_s, double &e1_t, double &e2_s, double &e2_t)
{

}
