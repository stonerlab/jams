// Copyright 2014 Joseph Barker. All rights reserved.

#include "solvers/metropolismc.h"

#include "core/consts.h"
#include "core/fields.h"
#include "core/globals.h"

void MetropolisMCSolver::initialize(int argc, char **argv, double idt) {
  using namespace globals;

    // initialize base class
  Solver::initialize(argc, argv, idt);

  output.write("Initialising Metropolis Monte-Carlo solver\n");

  output.write("  * Converting symmetric to general MAP matrices\n");

  J1ij_s.convertSymmetric2General();
  J1ij_t.convertSymmetric2General();
  J2ij_s.convertSymmetric2General();
  J2ij_t.convertSymmetric2General();

  output.write("  * Converting MAP to CSR\n");

  J1ij_s.convertMAP2CSR();
  J1ij_t.convertMAP2CSR();
  J2ij_s.convertMAP2CSR();
  J2ij_t.convertMAP2CSR();

  output.write("  * J1ij Scalar matrix memory (CSR): %f MB\n",
    J1ij_s.calculateMemory());
  output.write("  * J1ij Tensor matrix memory (CSR): %f MB\n",
    J1ij_t.calculateMemory());
  output.write("  * J2ij Scalar matrix memory (CSR): %f MB\n",
    J2ij_s.calculateMemory());
  output.write("  * J2ij Tensor matrix memory (CSR): %f MB\n",
    J2ij_t.calculateMemory());
  output.write("  * J4ijkl Scalar matrix memory (CSR): %f MB\n",
    J4ijkl_s.calculateMemoryUsage());

  initialized = true;
}

void MetropolisMCSolver::oneSpinEnergy(const int &i, double total[3]) {
  using namespace globals;

  total[0] = 0.0; total[1] = 0.0; total[2] = 0.0;

    // J1ij_s
  if (J1ij_s.nonZero() > 0) {
#ifdef CUDA
    const float *val = J1ij_s.valPtr();
#else
    const double *val = J1ij_s.valPtr();
#endif
    const int    *row = J1ij_s.rowPtr();
    const int    *indx = J1ij_s.colPtr();
    int           k;


    int begin = row[i]; int end = row[i+1];

        // upper triangle and diagonal
    for (int j = begin; j < end; ++j) {
      k = indx[j];
      for (int n = 0; n < 3; ++n) {
        total[n] -= s(k, n)*val[j];
      }
    }
  }

    if (J1ij_t.nonZero() > 0) {   // J1ij_t
#ifdef CUDA
      const float *val = J1ij_t.valPtr();
#else
      const double *val = J1ij_t.valPtr();
#endif
      const int    *row = J1ij_t.rowPtr();
      const int    *indx = J1ij_t.colPtr();
      const double *x   = s.data();
      int           k;


      for (int m = 0; m < 3; ++m) {
        int begin = row[3*i+m]; int end = row[3*i+m+1];

            // upper triangle and diagonal
        for (int j = begin; j < end; ++j) {
          k = indx[j];
          total[m] -= x[k]*val[j];
        }
      }
    }

    // J1ij_s
    if (J2ij_s.nonZero() > 0) {
#ifdef CUDA
      const float *val = J2ij_s.valPtr();
#else
      const double *val = J2ij_s.valPtr();
#endif
      const int    *row = J2ij_s.rowPtr();
      const int    *indx = J2ij_s.colPtr();
      int           k;


      int begin = row[i]; int end = row[i+1];

        // upper triangle and diagonal
      for (int j = begin; j < end; ++j) {
        k = indx[j];
        for (int n = 0; n < 3; ++n) {
          total[n] -= s(k, n)*val[j];
        }
      }
    }

    if (J2ij_t.nonZero() > 0) {   // J1ij_t
#ifdef CUDA
      const float *val = J2ij_t.valPtr();
#else
      const double *val = J2ij_t.valPtr();
#endif
      const int    *row = J2ij_t.rowPtr();
      const int    *indx = J2ij_t.colPtr();
      const double *x   = s.data();
      int           k;


      for (int m = 0; m < 3; ++m) {
        int begin = row[3*i+m]; int end = row[3*i+m+1];

            // upper triangle and diagonal
        for (int j = begin; j < end; ++j) {
          k = indx[j];
          total[m] -= x[k]*val[j];
        }
      }
    }

    //if(J4ijkl_s.nonZeros() > 0){ // J4ijkl_s
//#ifdef CUDA
        //const float *val = J4ijkl_s.valPtr();
//#else
        //const double *val = J4ijkl_s.valPtr();
//#endif
        //const int    *row = J4ijkl_s.pointersPtr();
        //const int    *coords = J4ijkl_s.cooPtr();


        //int begin = row[i]; int end = row[i+1];

        //// upper triangle and diagonal
        //for(int j = begin; j<end; ++j){
            //const int jidx = coords[3*j+0];
            //const int kidx = coords[3*j+1];
            //const int lidx = coords[3*j+2];

            //double sj[3], sk[3], sl[3];

            //for(int n=0; n<3; ++n){
                //sj[n] = s(jidx, n);
                //sk[n] = s(kidx, n);
                //sl[n] = s(lidx, n);
            //}

            //double k_dot_l = sk[0]*sl[0] + sk[1]*sl[1] + sk[2]*sl[2];
            //double j_dot_l = sj[0]*sl[0] + sj[1]*sl[1] + sj[2]*sl[2];
            //double j_dot_k = sk[0]*sj[0] + sk[1]*sj[1] + sk[2]*sj[2];

            //for(int n=0; n<3; ++n){
                //total[n] -= val[j]*(sj[n]*k_dot_l + sk[n]*j_dot_l + sl[n]*j_dot_k)/3.0;
            //}

        //}
    //}

  }

  void MetropolisMCSolver::run() {
    using namespace globals;

    const double theta = 0.1;
    const double Efactor = 0.671713067;  // muB/kB

    // pick spins randomly on average num_spins per step
    for (int n = 0; n < num_spins; ++n) {
      int i = rng.uniform()*(num_spins-1);

      double Enbr[3] = {0.0, 0.0, 0.0};
      oneSpinEnergy(i, Enbr);

      const double E1 = (Enbr[0]*s(i, 0) + Enbr[1]*s(i, 1) + Enbr[2]*s(i, 2));

        // trial move is random small angle
      double s_new[3];

      rng.sphere(s_new[0], s_new[1], s_new[2]);

      for (int j = 0; j < 3; ++j) {
        s_new[j] = s(i, j) + theta*s_new[j];
      }

        // normalise new spin
      const double norm =
        1.0/sqrt(s_new[0]*s_new[0] + s_new[1]*s_new[1] + s_new[2]*s_new[2]);
      for (int j = 0; j < 3; ++j) {
        s_new[j] = s_new[j]*norm;
      }

      const double E2 =
        (Enbr[0]*s_new[0] + Enbr[1]*s_new[1] + Enbr[2]*s_new[2]);

      double deltaE = E2-E1;

      if (deltaE < 0.0) {
        for (int j = 0; j < 3; ++j) {
          s(i, j) = s_new[j];
        }
      } else if (rng.uniform() < exp(-(deltaE*Efactor)/globalTemperature)) {
        for (int j = 0; j < 3; ++j) {
          s(i, j) = s_new[j];
        }
      }
    }

    for (int n = 0; n < num_spins; ++n) {
      int i = rng.uniform()*(num_spins-1);

      double Enbr[3] = {0.0, 0.0, 0.0};
      oneSpinEnergy(i, Enbr);

      const double E1 = (Enbr[0]*s(i, 0) + Enbr[1]*s(i, 1) + Enbr[2]*s(i, 2));

      // trial move is random small angle
      const double s_new[3]={-s(i, 0), -s(i, 1), -s(i, 2)};

      const double E2 =
        (Enbr[0]*s_new[0] + Enbr[1]*s_new[1] + Enbr[2]*s_new[2]);
      double deltaE = E2-E1;

      if (deltaE < 0.0) {
        for (int j = 0; j < 3; ++j) {
          s(i, j) = s_new[j];
        }
      } else if (rng.uniform() < exp(-(deltaE*Efactor)/globalTemperature)) {
        for (int j = 0; j < 3; ++j) {
          s(i, j) = s_new[j];
        }
      }
    }
    iteration++;
  }

  void MetropolisMCSolver::sync_device_data() {
  }

  void MetropolisMCSolver::compute_total_energy(double &e1_s, double &e1_t, double &e2_s, double &e2_t, double &e4_s) {
  }
