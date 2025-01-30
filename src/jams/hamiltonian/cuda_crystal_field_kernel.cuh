__global__ void cuda_crystal_field_energy_kernel(const unsigned int num_spins, const double * dev_s, const double* dev_cf_coeffs, double * dev_e) {
    const unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;

    if (idx < num_spins) {

        const double sx = dev_s[3 * idx + 0];
        const double sy = dev_s[3 * idx + 1];
        const double sz = dev_s[3 * idx + 2];

        double energy = 0.0;

      // C_{2,-2} Z_{2,-2}
      energy += dev_cf_coeffs[0*num_spins + idx] * 0.8660254037844386*(sx - 1.*sy)*(sx + sy);

// C_{2,-1} Z_{2,-1}
      energy += dev_cf_coeffs[1*num_spins + idx] * -1.7320508075688772*sx*sz;

// C_{2,0} Z_{2,0}
      energy += dev_cf_coeffs[2*num_spins + idx] * 0.5*(-1. + 3.*(sz*sz));

// C_{2,1} Z_{2,1}
      energy += dev_cf_coeffs[3*num_spins + idx] * -1.7320508075688772*sx*sz;

// C_{2,2} Z_{2,2}
      energy += dev_cf_coeffs[4*num_spins + idx] * 0.8660254037844386*(sx - 1.*sy)*(sx + sy);

// C_{4,-4} Z_{4,-4}
      energy += dev_cf_coeffs[5*num_spins + idx] * 0.739509972887452*(sx*sx*sx*sx - 6.*(sx*sx)*(sy*sy) + sy*sy*sy*sy);

// C_{4,-3} Z_{4,-3}
      energy += dev_cf_coeffs[6*num_spins + idx] * 2.091650066335189*sx*sz*(-1. + 4.*(sy*sy) + sz*sz);

// C_{4,-2} Z_{4,-2}
      energy += dev_cf_coeffs[7*num_spins + idx] * -0.5590169943749475*(-1. + 2.*(sy*sy) + sz*sz)*(-1. + 7.*(sz*sz));

// C_{4,-1} Z_{4,-1}
      energy += dev_cf_coeffs[8*num_spins + idx] * 0.7905694150420949*sx*sz*(3. - 7.*(sz*sz));

// C_{4,0} Z_{4,0}
      energy += dev_cf_coeffs[9*num_spins + idx] * 0.125*(3. - 30.*(sz*sz) + 35.*(sz*sz*sz*sz));

// C_{4,1} Z_{4,1}
      energy += dev_cf_coeffs[10*num_spins + idx] * 0.7905694150420949*sx*sz*(3. - 7.*(sz*sz));

// C_{4,2} Z_{4,2}
      energy += dev_cf_coeffs[11*num_spins + idx] * -0.5590169943749475*(-1. + 2.*(sy*sy) + sz*sz)*(-1. + 7.*(sz*sz));

// C_{4,3} Z_{4,3}
      energy += dev_cf_coeffs[12*num_spins + idx] * 2.091650066335189*sx*sz*(-1. + 4.*(sy*sy) + sz*sz);

// C_{4,4} Z_{4,4}
      energy += dev_cf_coeffs[13*num_spins + idx] * 0.739509972887452*(sx*sx*sx*sx - 6.*(sx*sx)*(sy*sy) + sy*sy*sy*sy);

// C_{6,-6} Z_{6,-6}
      energy += dev_cf_coeffs[14*num_spins + idx] * 0.6716932893813962*(sx*sx*sx*sx*sx*sx - 15.*(sx*sx*sx*sx)*(sy*sy) + 15.*(sx*sx)*(sy*sy*sy*sy) - 1.*(sy*sy*sy*sy*sy*sy));

// C_{6,-5} Z_{6,-5}
      energy += dev_cf_coeffs[15*num_spins + idx] * -2.3268138086232857*sx*(sx*sx*sx*sx - 10.*(sx*sx)*(sy*sy) + 5.*(sy*sy*sy*sy))*sz;

// C_{6,-4} Z_{6,-4}
      energy += dev_cf_coeffs[16*num_spins + idx] * 0.49607837082461076*(sx*sx*sx*sx - 6.*(sx*sx)*(sy*sy) + sy*sy*sy*sy)*(-1. + 11.*(sz*sz));

// C_{6,-3} Z_{6,-3}
      energy += dev_cf_coeffs[17*num_spins + idx] * 0.9057110466368399*sx*sz*(-1. + 4.*(sy*sy) + sz*sz)*(-3. + 11.*(sz*sz));

// C_{6,-2} Z_{6,-2}
      energy += dev_cf_coeffs[18*num_spins + idx] * -0.45285552331841994*(-1. + 2.*(sy*sy) + sz*sz)*(1. - 18.*(sz*sz) + 33.*(sz*sz*sz*sz));

// C_{6,-1} Z_{6,-1}
      energy += dev_cf_coeffs[19*num_spins + idx] * -0.57282196186948*sx*sz*(5. - 30.*(sz*sz) + 33.*(sz*sz*sz*sz));

// C_{6,0} Z_{6,0}
      energy += dev_cf_coeffs[20*num_spins + idx] * 0.0625*(-5. + 21.*(sz*sz)*(5. - 15.*(sz*sz) + 11.*(sz*sz*sz*sz)));

// C_{6,1} Z_{6,1}
      energy += dev_cf_coeffs[21*num_spins + idx] * -0.57282196186948*sx*sz*(5. - 30.*(sz*sz) + 33.*(sz*sz*sz*sz));

// C_{6,2} Z_{6,2}
      energy += dev_cf_coeffs[22*num_spins + idx] * -0.45285552331841994*(-1. + 2.*(sy*sy) + sz*sz)*(1. - 18.*(sz*sz) + 33.*(sz*sz*sz*sz));

// C_{6,3} Z_{6,3}
      energy += dev_cf_coeffs[23*num_spins + idx] * 0.9057110466368399*sx*sz*(-1. + 4.*(sy*sy) + sz*sz)*(-3. + 11.*(sz*sz));

// C_{6,4} Z_{6,4}
      energy += dev_cf_coeffs[24*num_spins + idx] * 0.49607837082461076*(sx*sx*sx*sx - 6.*(sx*sx)*(sy*sy) + sy*sy*sy*sy)*(-1. + 11.*(sz*sz));

// C_{6,5} Z_{6,5}
      energy += dev_cf_coeffs[25*num_spins + idx] * -2.3268138086232857*sx*(sx*sx*sx*sx - 10.*(sx*sx)*(sy*sy) + 5.*(sy*sy*sy*sy))*sz;

// C_{6,6} Z_{6,6}
      energy += dev_cf_coeffs[26*num_spins + idx] * 0.6716932893813962*(sx*sx*sx*sx*sx*sx - 15.*(sx*sx*sx*sx)*(sy*sy) + 15.*(sx*sx)*(sy*sy*sy*sy) - 1.*(sy*sy*sy*sy*sy*sy));

      dev_e[idx] = energy;

    }
}


__global__ void cuda_crystal_field_kernel(const unsigned int num_spins, const double * dev_s, const double* dev_cf_coeffs, double * dev_h) {
    const unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;

    if (idx < num_spins) {

        double sx = dev_s[3*idx + 0];
        double sy = dev_s[3*idx + 1];
        double sz = dev_s[3*idx + 2];

        double h[3] = {0, 0, 0};

// -C_{2,-2} dZ_{2,-2}/dS
      double C2_2 = dev_cf_coeffs[0*num_spins + idx];
      h[0] += C2_2*( -1.7320508075688772*sx*(2.*(sy*sy) + sz*sz) );
      h[1] += C2_2*( -1.7320508075688772*sy*(-2. + 2.*(sy*sy) + sz*sz) );
      h[2] += C2_2*( -1.7320508075688772*sz*(-1. + 2.*(sy*sy) + sz*sz) );

// -C_{2,-1} dZ_{2,-1}/dS
      double C2_1 = dev_cf_coeffs[1*num_spins + idx];
      h[0] += C2_1*( 1.7320508075688772*sz*(-1. + 2.*(sy*sy) + 2.*(sz*sz)) );
      h[1] += C2_1*( -3.4641016151377544*sx*sy*sz );
      h[2] += C2_1*( -1.7320508075688772*sx*(-1. + 2.*(sz*sz)) );

// -C_{2,0} dZ_{2,0}/dS
      double C20 = dev_cf_coeffs[2*num_spins + idx];
      h[0] += C20*( 3.*sx*(sz*sz) );
      h[1] += C20*( 3.*sy*(sz*sz) );
      h[2] += C20*( 3.*sz*(-1. + sz*sz) );

// -C_{2,1} dZ_{2,1}/dS
      double C21 = dev_cf_coeffs[3*num_spins + idx];
      h[0] += C21*( 1.7320508075688772*sz*(-1. + 2.*(sy*sy) + 2.*(sz*sz)) );
      h[1] += C21*( -3.4641016151377544*sx*sy*sz );
      h[2] += C21*( -1.7320508075688772*sx*(-1. + 2.*(sz*sz)) );

// -C_{2,2} dZ_{2,2}/dS
      double C22 = dev_cf_coeffs[4*num_spins + idx];
      h[0] += C22*( -1.7320508075688772*sx*(2.*(sy*sy) + sz*sz) );
      h[1] += C22*( -1.7320508075688772*sy*(-2. + 2.*(sy*sy) + sz*sz) );
      h[2] += C22*( -1.7320508075688772*sz*(-1. + 2.*(sy*sy) + sz*sz) );

// -C_{4,-4} dZ_{4,-4}/dS
      double C4_4 = dev_cf_coeffs[5*num_spins + idx];
      h[0] += C4_4*( 2.958039891549808*sx*(8.*(sy*sy*sy*sy) - 1.*(sz*sz) + sz*sz*sz*sz + sy*sy*(-4. + 8.*(sz*sz))) );
      h[1] += C4_4*( 2.958039891549808*sy*(4. + 8.*(sy*sy*sy*sy) - 5.*(sz*sz) + sz*sz*sz*sz + 4.*(sy*sy)*(-3. + 2.*(sz*sz))) );
      h[2] += C4_4*( 2.958039891549808*(sx*sx*sx*sx - 6.*(sx*sx)*(sy*sy) + sy*sy*sy*sy)*sz );

// -C_{4,-3} dZ_{4,-3}/dS
      double C4_3 = dev_cf_coeffs[6*num_spins + idx];
      h[0] += C4_3*( -2.091650066335189*sz*(1. + 16.*(sy*sy*sy*sy) - 5.*(sz*sz) + 4.*(sz*sz*sz*sz) + 2.*(sy*sy)*(-7. + 10.*(sz*sz))) );
      h[1] += C4_3*( 4.183300132670378*sx*sy*sz*(-5. + 8.*(sy*sy) + 2.*(sz*sz)) );
      h[2] += C4_3*( 2.091650066335189*sx*(-1. + 4.*(sy*sy) + sz*sz)*(-1. + 4.*(sz*sz)) );

// -C_{4,-2} dZ_{4,-2}/dS
      double C4_2 = dev_cf_coeffs[7*num_spins + idx];
      h[0] += C4_2*( -2.23606797749979*sx*(-1.*(sy*sy) + 2.*(-2. + 7.*(sy*sy))*(sz*sz) + 7.*(sz*sz*sz*sz)) );
      h[1] += C4_2*( -2.23606797749979*sy*(1. - 11.*(sz*sz) + 7.*(sz*sz*sz*sz) + sy*sy*(-1. + 14.*(sz*sz))) );
      h[2] += C4_2*( -2.23606797749979*sz*(-1. + 2.*(sy*sy) + sz*sz)*(-4. + 7.*(sz*sz)) );

// -C_{4,-1} dZ_{4,-1}/dS
      double C4_1 = dev_cf_coeffs[8*num_spins + idx];
      h[0] += C4_1*( 0.7905694150420949*sz*(3. - 27.*(sz*sz) + 28.*(sz*sz*sz*sz) + sy*sy*(-6. + 28.*(sz*sz))) );
      h[1] += C4_1*( -1.5811388300841898*sx*sy*sz*(-3. + 14.*(sz*sz)) );
      h[2] += C4_1*( -0.7905694150420949*sx*(3. - 27.*(sz*sz) + 28.*(sz*sz*sz*sz)) );

// -C_{4,0} dZ_{4,0}/dS
      double C40 = dev_cf_coeffs[9*num_spins + idx];
      h[0] += C40*( -2.5*sx*(sz*sz)*(3. - 7.*(sz*sz)) );
      h[1] += C40*( -2.5*sy*(sz*sz)*(3. - 7.*(sz*sz)) );
      h[2] += C40*( 2.5*sz*(3. - 10.*(sz*sz) + 7.*(sz*sz*sz*sz)) );

// -C_{4,1} dZ_{4,1}/dS
      double C41 = dev_cf_coeffs[10*num_spins + idx];
      h[0] += C41*( 0.7905694150420949*sz*(3. - 27.*(sz*sz) + 28.*(sz*sz*sz*sz) + sy*sy*(-6. + 28.*(sz*sz))) );
      h[1] += C41*( -1.5811388300841898*sx*sy*sz*(-3. + 14.*(sz*sz)) );
      h[2] += C41*( -0.7905694150420949*sx*(3. - 27.*(sz*sz) + 28.*(sz*sz*sz*sz)) );

// -C_{4,2} dZ_{4,2}/dS
      double C42 = dev_cf_coeffs[11*num_spins + idx];
      h[0] += C42*( -2.23606797749979*sx*(-1.*(sy*sy) + 2.*(-2. + 7.*(sy*sy))*(sz*sz) + 7.*(sz*sz*sz*sz)) );
      h[1] += C42*( -2.23606797749979*sy*(1. - 11.*(sz*sz) + 7.*(sz*sz*sz*sz) + sy*sy*(-1. + 14.*(sz*sz))) );
      h[2] += C42*( -2.23606797749979*sz*(-1. + 2.*(sy*sy) + sz*sz)*(-4. + 7.*(sz*sz)) );

// -C_{4,3} dZ_{4,3}/dS
      double C43 = dev_cf_coeffs[12*num_spins + idx];
      h[0] += C43*( -2.091650066335189*sz*(1. + 16.*(sy*sy*sy*sy) - 5.*(sz*sz) + 4.*(sz*sz*sz*sz) + 2.*(sy*sy)*(-7. + 10.*(sz*sz))) );
      h[1] += C43*( 4.183300132670378*sx*sy*sz*(-5. + 8.*(sy*sy) + 2.*(sz*sz)) );
      h[2] += C43*( 2.091650066335189*sx*(-1. + 4.*(sy*sy) + sz*sz)*(-1. + 4.*(sz*sz)) );

// -C_{4,4} dZ_{4,4}/dS
      double C44 = dev_cf_coeffs[13*num_spins + idx];
      h[0] += C44*( 2.958039891549808*sx*(8.*(sy*sy*sy*sy) - 1.*(sz*sz) + sz*sz*sz*sz + sy*sy*(-4. + 8.*(sz*sz))) );
      h[1] += C44*( 2.958039891549808*sy*(4. + 8.*(sy*sy*sy*sy) - 5.*(sz*sz) + sz*sz*sz*sz + 4.*(sy*sy)*(-3. + 2.*(sz*sz))) );
      h[2] += C44*( 2.958039891549808*(sx*sx*sx*sx - 6.*(sx*sx)*(sy*sy) + sy*sy*sy*sy)*sz );

// -C_{6,-6} dZ_{6,-6}/dS
      double C6_6 = dev_cf_coeffs[14*num_spins + idx];
      h[0] += C6_6*( -4.030159736288377*sx*(6.*(sy*sy*sy*sy*sy*sy) + 5.*(sy*sy*sy*sy)*(sz*sz) - 10.*(sx*sx)*(sy*sy)*(2.*(sy*sy) + sz*sz) + sx*sx*sx*sx*(6.*(sy*sy) + sz*sz)) );
      h[1] += C6_6*( 4.030159736288377*sy*(6.*(sx*sx*sx*sx*sx*sx) - 20.*(sx*sx*sx*sx)*(sy*sy) + 6.*(sx*sx)*(sy*sy*sy*sy) + (5.*(sx*sx*sx*sx) - 10.*(sx*sx)*(sy*sy) + sy*sy*sy*sy)*(sz*sz)) );
      h[2] += C6_6*( 4.030159736288377*(sx*sx*sx*sx*sx*sx - 15.*(sx*sx*sx*sx)*(sy*sy) + 15.*(sx*sx)*(sy*sy*sy*sy) - 1.*(sy*sy*sy*sy*sy*sy))*sz );

// -C_{6,-5} dZ_{6,-5}/dS
      double C6_5 = dev_cf_coeffs[15*num_spins + idx];
      h[0] += C6_5*( 2.3268138086232857*sz*(-1.*(sx*sx*sx*sx*sx*sx) + 35.*(sx*sx*sx*sx)*(sy*sy) - 55.*(sx*sx)*(sy*sy*sy*sy) + 5.*(sy*sy*sy*sy*sy*sy) + 5.*(sx*sx*sx*sx - 6.*(sx*sx)*(sy*sy) + sy*sy*sy*sy)*(sz*sz)) );
      h[1] += C6_5*( -4.653627617246571*sx*sy*sz*(13. - 56.*(sy*sy) + 48.*(sy*sy*sy*sy) + 4.*(-4. + 9.*(sy*sy))*(sz*sz) + 3.*(sz*sz*sz*sz)) );
      h[2] += C6_5*( -2.3268138086232857*sx*(sx*sx*sx*sx - 10.*(sx*sx)*(sy*sy) + 5.*(sy*sy*sy*sy))*(-1. + 6.*(sz*sz)) );

// -C_{6,-4} dZ_{6,-4}/dS
      double C6_4 = dev_cf_coeffs[16*num_spins + idx];
      h[0] += C6_4*( 0.9921567416492215*sx*(13.*(sz*sz) - 46.*(sz*sz*sz*sz) + 33.*(sz*sz*sz*sz*sz*sz) + 8.*(sy*sy*sy*sy)*(-2. + 33.*(sz*sz)) + 8.*(sy*sy)*(1. - 24.*(sz*sz) + 33.*(sz*sz*sz*sz))) );
      h[1] += C6_4*( 0.9921567416492215*sy*(-8. + 109.*(sz*sz) - 134.*(sz*sz*sz*sz) + 33.*(sz*sz*sz*sz*sz*sz) + 8.*(sy*sy*sy*sy)*(-2. + 33.*(sz*sz)) + 8.*(sy*sy)*(3. - 46.*(sz*sz) + 33.*(sz*sz*sz*sz))) );
      h[2] += C6_4*( -0.9921567416492215*(sx*sx*sx*sx - 6.*(sx*sx)*(sy*sy) + sy*sy*sy*sy)*sz*(13. - 33.*(sz*sz)) );

// -C_{6,-3} dZ_{6,-3}/dS
      double C6_3 = dev_cf_coeffs[17*num_spins + idx];
      h[0] += C6_3*( -2.7171331399105196*sz*(-1. + 16.*(sz*sz) - 37.*(sz*sz*sz*sz) + 22.*(sz*sz*sz*sz*sz*sz) + 8.*(sy*sy*sy*sy)*(-2. + 11.*(sz*sz)) + 2.*(sy*sy)*(7. - 54.*(sz*sz) + 55.*(sz*sz*sz*sz))) );
      h[1] += C6_3*( 5.434266279821039*sx*sy*sz*(5. - 24.*(sz*sz) + 11.*(sz*sz*sz*sz) + sy*sy*(-8. + 44.*(sz*sz))) );
      h[2] += C6_3*( 2.7171331399105196*sx*(-1. + 4.*(sy*sy) + sz*sz)*(1. - 15.*(sz*sz) + 22.*(sz*sz*sz*sz)) );

// -C_{6,-2} dZ_{6,-2}/dS
      double C6_2 = dev_cf_coeffs[18*num_spins + idx];
      h[0] += C6_2*( -0.9057110466368399*sx*(2.*(sy*sy) + (19. - 72.*(sy*sy))*(sz*sz) + 6.*(-17. + 33.*(sy*sy))*(sz*sz*sz*sz) + 99.*(sz*sz*sz*sz*sz*sz)) );
      h[1] += C6_2*( -0.9057110466368399*sy*(-2. + 55.*(sz*sz) - 168.*(sz*sz*sz*sz) + 99.*(sz*sz*sz*sz*sz*sz) + 2.*(sy*sy)*(1. - 36.*(sz*sz) + 99.*(sz*sz*sz*sz))) );
      h[2] += C6_2*( -0.9057110466368399*sz*(-1. + 2.*(sy*sy) + sz*sz)*(19. - 102.*(sz*sz) + 99.*(sz*sz*sz*sz)) );

// -C_{6,-1} dZ_{6,-1}/dS
      double C6_1 = dev_cf_coeffs[19*num_spins + idx];
      h[0] += C6_1*( -0.57282196186948*sz*(5. - 100.*(sz*sz) + 285.*(sz*sz*sz*sz) - 198.*(sz*sz*sz*sz*sz*sz) - 2.*(sy*sy)*(5. - 60.*(sz*sz) + 99.*(sz*sz*sz*sz))) );
      h[1] += C6_1*( -1.14564392373896*sx*sy*sz*(5. - 60.*(sz*sz) + 99.*(sz*sz*sz*sz)) );
      h[2] += C6_1*( -0.57282196186948*sx*(-5. + 100.*(sz*sz) - 285.*(sz*sz*sz*sz) + 198.*(sz*sz*sz*sz*sz*sz)) );

// -C_{6,0} dZ_{6,0}/dS
      double C60 = dev_cf_coeffs[20*num_spins + idx];
      h[0] += C60*( 2.625*sx*(sz*sz)*(5. - 30.*(sz*sz) + 33.*(sz*sz*sz*sz)) );
      h[1] += C60*( 2.625*sy*(sz*sz)*(5. - 30.*(sz*sz) + 33.*(sz*sz*sz*sz)) );
      h[2] += C60*( 2.625*sz*(-5. + 35.*(sz*sz) - 63.*(sz*sz*sz*sz) + 33.*(sz*sz*sz*sz*sz*sz)) );

// -C_{6,1} dZ_{6,1}/dS
      double C61 = dev_cf_coeffs[21*num_spins + idx];
      h[0] += C61*( -0.57282196186948*sz*(5. - 100.*(sz*sz) + 285.*(sz*sz*sz*sz) - 198.*(sz*sz*sz*sz*sz*sz) - 2.*(sy*sy)*(5. - 60.*(sz*sz) + 99.*(sz*sz*sz*sz))) );
      h[1] += C61*( -1.14564392373896*sx*sy*sz*(5. - 60.*(sz*sz) + 99.*(sz*sz*sz*sz)) );
      h[2] += C61*( -0.57282196186948*sx*(-5. + 100.*(sz*sz) - 285.*(sz*sz*sz*sz) + 198.*(sz*sz*sz*sz*sz*sz)) );

// -C_{6,2} dZ_{6,2}/dS
      double C62 = dev_cf_coeffs[22*num_spins + idx];
      h[0] += C62*( -0.9057110466368399*sx*(2.*(sy*sy) + (19. - 72.*(sy*sy))*(sz*sz) + 6.*(-17. + 33.*(sy*sy))*(sz*sz*sz*sz) + 99.*(sz*sz*sz*sz*sz*sz)) );
      h[1] += C62*( -0.9057110466368399*sy*(-2. + 55.*(sz*sz) - 168.*(sz*sz*sz*sz) + 99.*(sz*sz*sz*sz*sz*sz) + 2.*(sy*sy)*(1. - 36.*(sz*sz) + 99.*(sz*sz*sz*sz))) );
      h[2] += C62*( -0.9057110466368399*sz*(-1. + 2.*(sy*sy) + sz*sz)*(19. - 102.*(sz*sz) + 99.*(sz*sz*sz*sz)) );

// -C_{6,3} dZ_{6,3}/dS
      double C63 = dev_cf_coeffs[23*num_spins + idx];
      h[0] += C63*( -2.7171331399105196*sz*(-1. + 16.*(sz*sz) - 37.*(sz*sz*sz*sz) + 22.*(sz*sz*sz*sz*sz*sz) + 8.*(sy*sy*sy*sy)*(-2. + 11.*(sz*sz)) + 2.*(sy*sy)*(7. - 54.*(sz*sz) + 55.*(sz*sz*sz*sz))) );
      h[1] += C63*( 5.434266279821039*sx*sy*sz*(5. - 24.*(sz*sz) + 11.*(sz*sz*sz*sz) + sy*sy*(-8. + 44.*(sz*sz))) );
      h[2] += C63*( 2.7171331399105196*sx*(-1. + 4.*(sy*sy) + sz*sz)*(1. - 15.*(sz*sz) + 22.*(sz*sz*sz*sz)) );

// -C_{6,4} dZ_{6,4}/dS
      double C64 = dev_cf_coeffs[24*num_spins + idx];
      h[0] += C64*( 0.9921567416492215*sx*(13.*(sz*sz) - 46.*(sz*sz*sz*sz) + 33.*(sz*sz*sz*sz*sz*sz) + 8.*(sy*sy*sy*sy)*(-2. + 33.*(sz*sz)) + 8.*(sy*sy)*(1. - 24.*(sz*sz) + 33.*(sz*sz*sz*sz))) );
      h[1] += C64*( 0.9921567416492215*sy*(-8. + 109.*(sz*sz) - 134.*(sz*sz*sz*sz) + 33.*(sz*sz*sz*sz*sz*sz) + 8.*(sy*sy*sy*sy)*(-2. + 33.*(sz*sz)) + 8.*(sy*sy)*(3. - 46.*(sz*sz) + 33.*(sz*sz*sz*sz))) );
      h[2] += C64*( -0.9921567416492215*(sx*sx*sx*sx - 6.*(sx*sx)*(sy*sy) + sy*sy*sy*sy)*sz*(13. - 33.*(sz*sz)) );

// -C_{6,5} dZ_{6,5}/dS
      double C65 = dev_cf_coeffs[25*num_spins + idx];
      h[0] += C65*( 2.3268138086232857*sz*(-1.*(sx*sx*sx*sx*sx*sx) + 35.*(sx*sx*sx*sx)*(sy*sy) - 55.*(sx*sx)*(sy*sy*sy*sy) + 5.*(sy*sy*sy*sy*sy*sy) + 5.*(sx*sx*sx*sx - 6.*(sx*sx)*(sy*sy) + sy*sy*sy*sy)*(sz*sz)) );
      h[1] += C65*( -4.653627617246571*sx*sy*sz*(13. - 56.*(sy*sy) + 48.*(sy*sy*sy*sy) + 4.*(-4. + 9.*(sy*sy))*(sz*sz) + 3.*(sz*sz*sz*sz)) );
      h[2] += C65*( -2.3268138086232857*sx*(sx*sx*sx*sx - 10.*(sx*sx)*(sy*sy) + 5.*(sy*sy*sy*sy))*(-1. + 6.*(sz*sz)) );

// -C_{6,6} dZ_{6,6}/dS
      double C66 = dev_cf_coeffs[26*num_spins + idx];
      h[0] += C66*( -4.030159736288377*sx*(6.*(sy*sy*sy*sy*sy*sy) + 5.*(sy*sy*sy*sy)*(sz*sz) - 10.*(sx*sx)*(sy*sy)*(2.*(sy*sy) + sz*sz) + sx*sx*sx*sx*(6.*(sy*sy) + sz*sz)) );
      h[1] += C66*( 4.030159736288377*sy*(6.*(sx*sx*sx*sx*sx*sx) - 20.*(sx*sx*sx*sx)*(sy*sy) + 6.*(sx*sx)*(sy*sy*sy*sy) + (5.*(sx*sx*sx*sx) - 10.*(sx*sx)*(sy*sy) + sy*sy*sy*sy)*(sz*sz)) );
      h[2] += C66*( 4.030159736288377*(sx*sx*sx*sx*sx*sx - 15.*(sx*sx*sx*sx)*(sy*sy) + 15.*(sx*sx)*(sy*sy*sy*sy) - 1.*(sy*sy*sy*sy*sy*sy))*sz );


        dev_h[3*idx + 0] = h[0];
        dev_h[3*idx + 1] = h[1];
        dev_h[3*idx + 2] = h[2];

    }
}