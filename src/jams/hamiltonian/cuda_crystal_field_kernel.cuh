
__global__ void cuda_crystal_field_kernel(const unsigned int num_spins, const double * dev_s, const double* dev_cf_coeffs) {
    const unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;

    if (idx < num_spins) {

        double sx = dev_s[3*idx + 0];
        double sy = dev_s[3*idx + 1];
        double sz = dev_s[3*idx + 2];

        double sx2 = sx*sx;
        double sy2 = sy*sy;
        double sz2 = sz*sz;

        double sx3 = sx2*sx;
        double sy3 = sy2*sy;
        double sz3 = sz2*sz;

        double sx4 = sx2*sx2;
        double sy4 = sy2*sy2;
        double sz4 = sz2*sz2;

        double sx5 = sx3*sx2;
        double sy5 = sy3*sy2;
        double sz5 = sz3*sz2;

        double sx6 = sx4*sx2;
        double sy6 = sy4*sy2;
        double sz6 = sz4*sz2;

        double h[3] = {0, 0, 0};

// -C_{2,-2} dZ_{2,-2}/dS

        double C2_2 = dev_cf_coeffs[0*num_spins + idx];

        h[0] += C2_2*( -6.*sy*(-1. + 2.*sy2 + 2.*sz2) );
        h[1] += C2_2*( 6.*sx*(-1. + 2.*sy2) );
        h[2] += C2_2*( 12.*sx*sy*sz );

// -C_{2,-1} dZ_{2,-1}/dS

        double C2_1 = dev_cf_coeffs[1*num_spins + idx];

        h[0] += C2_1*( -6.*sx*sy*sz );
        h[1] += C2_1*( 3.*(1. - 2.*sy2)*sz );
        h[2] += C2_1*( 3.*sy*(1. - 2.*sz2) );

// -C_{2,0} dZ_{2,0}/dS

        double C20 = dev_cf_coeffs[2*num_spins + idx];

        h[0] += C20*( 3.*sx*sz2 );
        h[1] += C20*( 3.*sy*sz2 );
        h[2] += C20*( 3.*sz*(-1. + sz2) );

// -C_{2,1} dZ_{2,1}/dS

        double C21 = dev_cf_coeffs[3*num_spins + idx];

        h[0] += C21*( 3.*sz*(-1. + 2.*sy2 + 2.*sz2) );
        h[1] += C21*( -6.*sx*sy*sz );
        h[2] += C21*( 3.*sx*(1. - 2.*sz2) );

// -C_{2,2} dZ_{2,2}/dS

        double C22 = dev_cf_coeffs[4*num_spins + idx];

        h[0] += C22*( -6.*sx*(2.*sy2 + sz2) );
        h[1] += C22*( 6.*sy*(2.*sx2 + sz2) );
        h[2] += C22*( 6.*(sx2 - 1.*sy2)*sz );

// -C_{4,-4} dZ_{4,-4}/dS

        double C4_4 = dev_cf_coeffs[5*num_spins + idx];

        h[0] += C4_4*( 420.*sy*(1. + 8.*sy4 - 5.*sz2 + 4.*sy2*(-2. + 3.*sz2) + 4.*sz4) );
        h[1] += C4_4*( -420.*sx*(1. + 8.*sy4 + 4.*sy2*(-2. + sz2) - 1.*sz2) );
        h[2] += C4_4*( 1680.*sx*sy*(sx2 - 1.*sy2)*sz );

// -C_{4,-3} dZ_{4,-3}/dS

        double C4_3 = dev_cf_coeffs[6*num_spins + idx];

        h[0] += C4_3*( 210.*sx*sy*sz*(-3. + 8.*sy2 + 6.*sz2) );
        h[1] += C4_3*( 105.*sz*(3. - 3.*sz2 + 2.*sy2*(-9. + 8.*sy2 + 6.*sz2)) );
        h[2] += C4_3*( 105.*sy*(-3. + 4.*sy2 + 3.*sz2)*(-1. + 4.*sz2) );

// -C_{4,-2} dZ_{4,-2}/dS

        double C4_2 = dev_cf_coeffs[7*num_spins + idx];

        h[0] += C4_2*( -15.*sy*(1. - 23.*sz2 + sy2*(-2. + 28.*sz2) + 28.*sz4) );
        h[1] += C4_2*( 15.*sx*(1. - 7.*sz2 + sy2*(-2. + 28.*sz2)) );
        h[2] += C4_2*( 60.*sx*sy*sz*(-4. + 7.*sz2) );

// -C_{4,-1} dZ_{4,-1}/dS

        double C4_1 = dev_cf_coeffs[8*num_spins + idx];

        h[0] += C4_1*( 5.*sx*sy*sz*(3. - 14.*sz2) );
        h[1] += C4_1*( -2.5*sz*(3. - 7.*sz2 + sy2*(-6. + 28.*sz2)) );
        h[2] += C4_1*( -2.5*sy*(3. - 27.*sz2 + 28.*sz4) );

// -C_{4,0} dZ_{4,0}/dS

        double C40 = dev_cf_coeffs[9*num_spins + idx];

        h[0] += C40*( 2.5*sx*sz2*(-3. + 7.*sz2) );
        h[1] += C40*( 2.5*sy*sz2*(-3. + 7.*sz2) );
        h[2] += C40*( 2.5*sz*(3. - 10.*sz2 + 7.*sz4) );

// -C_{4,1} dZ_{4,1}/dS

        double C41 = dev_cf_coeffs[10*num_spins + idx];

        h[0] += C41*( 2.5*sz*(3. - 27.*sz2 + sy2*(-6. + 28.*sz2) + 28.*sz4) );
        h[1] += C41*( 5.*sx*sy*sz*(3. - 14.*sz2) );
        h[2] += C41*( -2.5*sx*(3. - 27.*sz2 + 28.*sz4) );

// -C_{4,2} dZ_{4,2}/dS

        double C42 = dev_cf_coeffs[11*num_spins + idx];

        h[0] += C42*( -30.*sx*(-1.*sy2 + 2.*(-2. + 7.*sy2)*sz2 + 7.*sz4) );
        h[1] += C42*( -30.*sy*(1. - 11.*sz2 + sy2*(-1. + 14.*sz2) + 7.*sz4) );
        h[2] += C42*( -30.*sz*(-1. + 2.*sy2 + sz2)*(-4. + 7.*sz2) );

// -C_{4,3} dZ_{4,3}/dS

        double C43 = dev_cf_coeffs[12*num_spins + idx];

        h[0] += C43*( -105.*sz*(1. + 16.*sy4 - 5.*sz2 + 2.*sy2*(-7. + 10.*sz2) + 4.*sz4) );
        h[1] += C43*( 210.*sx*sy*sz*(-5. + 8.*sy2 + 2.*sz2) );
        h[2] += C43*( 105.*sx*(-1. + 4.*sy2 + sz2)*(-1. + 4.*sz2) );

// -C_{4,4} dZ_{4,4}/dS

        double C44 = dev_cf_coeffs[13*num_spins + idx];

        h[0] += C44*( 420.*sx*(8.*sy4 - 1.*sz2 + sy2*(-4. + 8.*sz2) + sz4) );
        h[1] += C44*( 420.*sy*(4. + 8.*sy4 - 5.*sz2 + 4.*sy2*(-3. + 2.*sz2) + sz4) );
        h[2] += C44*( 420.*(sx4 - 6.*sx2*sy2 + sy4)*sz );

// -C_{6,-6} dZ_{6,-6}/dS

        double C6_6 = dev_cf_coeffs[14*num_spins + idx];

        h[0] += C6_6*( -62370.*sy*(-1.*sx6 + sy4*(sy2 + sz2) + 5.*sx4*(3.*sy2 + sz2) - 5.*sx2*(3.*sy4 + 2.*sy2*sz2)) );
        h[1] += C6_6*( 62370.*sx*(32.*sy6 - 1.*pow(-1. + sz2,2) + 16.*sy4*(-3. + 2.*sz2) + 6.*sy2*(3. - 4.*sz2 + sz4)) );
        h[2] += C6_6*( 124740.*(3.*sx5*sy - 10.*sx3*sy3 + 3.*sx*sy5)*sz );

// -C_{6,-5} dZ_{6,-5}/dS

        double C6_5 = dev_cf_coeffs[15*num_spins + idx];

        h[0] += C6_5*( -20790.*sx*sy*sz*(5. - 40.*sy2 + 48.*sy4 + 20.*(-1. + 3.*sy2)*sz2 + 15.*sz4) );
        h[1] += C6_5*( -10395.*sz*(96.*sy6 - 5.*pow(-1. + sz2,2) + 40.*sy4*(-4. + 3.*sz2) + 10.*sy2*(7. - 10.*sz2 + 3.*sz4)) );
        h[2] += C6_5*( -10395.*sy*(16.*sy4 + 20.*sy2*(-1. + sz2) + 5.*pow(-1. + sz2,2))*(-1. + 6.*sz2) );

// -C_{6,-4} dZ_{6,-4}/dS

        double C6_4 = dev_cf_coeffs[16*num_spins + idx];

        h[0] += C6_4*( 1890.*sy*(-1. + 38.*sz2 + 4.*sy4*(-2. + 33.*sz2) - 103.*sz4 + 2.*sy2*(4. - 83.*sz2 + 99.*sz4) + 66.*sz6) );
        h[1] += C6_4*( -1890.*sx*(-1. + 8.*sy2 - 8.*sy4 + 6.*(2. - 19.*sy2 + 22.*sy4)*sz2 + 11.*(-1. + 6.*sy2)*sz4) );
        h[2] += C6_4*( -3780.*sx*sy*sz*(-1. + 2.*sy2 + sz2)*(-13. + 33.*sz2) );

// -C_{6,-3} dZ_{6,-3}/dS

        double C6_3 = dev_cf_coeffs[17*num_spins + idx];

        h[0] += C6_3*( 945.*sx*sy*sz*(3. - 28.*sz2 + sy2*(-8. + 44.*sz2) + 33.*sz4) );
        h[1] += C6_3*( 472.5*sz*(-3. + 14.*sz2 + 8.*sy4*(-2. + 11.*sz2) - 11.*sz4 + 2.*sy2*(9. - 50.*sz2 + 33.*sz4)) );
        h[2] += C6_3*( 472.5*sy*(-3. + 4.*sy2 + 3.*sz2)*(1. - 15.*sz2 + 22.*sz4) );

// -C_{6,-2} dZ_{6,-2}/dS

        double C6_2 = dev_cf_coeffs[18*num_spins + idx];

        h[0] += C6_2*( -26.25*sy*(-1. + 2.*sy2 + 8.*(7. - 9.*sy2)*sz2 + 3.*(-79. + 66.*sy2)*sz4 + 198.*sz6) );
        h[1] += C6_2*( 26.25*sx*(-1. + 18.*sz2 - 33.*sz4 + 2.*sy2*(1. - 36.*sz2 + 99.*sz4)) );
        h[2] += C6_2*( 52.5*sx*sy*sz*(19. - 102.*sz2 + 99.*sz4) );

// -C_{6,-1} dZ_{6,-1}/dS

        double C6_1 = dev_cf_coeffs[19*num_spins + idx];

        h[0] += C6_1*( -5.25*sx*sy*sz*(5. - 60.*sz2 + 99.*sz4) );
        h[1] += C6_1*( 2.625*sz*(5. - 30.*sz2 + 33.*sz4 - 2.*sy2*(5. - 60.*sz2 + 99.*sz4)) );
        h[2] += C6_1*( -2.625*sy*(-5. + 100.*sz2 - 285.*sz4 + 198.*sz6) );

// -C_{6,0} dZ_{6,0}/dS

        double C60 = dev_cf_coeffs[20*num_spins + idx];

        h[0] += C60*( 2.625*sx*sz2*(5. - 30.*sz2 + 33.*sz4) );
        h[1] += C60*( 2.625*sy*sz2*(5. - 30.*sz2 + 33.*sz4) );
        h[2] += C60*( 2.625*sz*(-5. + 35.*sz2 - 63.*sz4 + 33.*sz6) );

// -C_{6,1} dZ_{6,1}/dS

        double C61 = dev_cf_coeffs[21*num_spins + idx];

        h[0] += C61*( 2.625*sz*(-5. + 100.*sz2 - 285.*sz4 + 2.*sy2*(5. - 60.*sz2 + 99.*sz4) + 198.*sz6) );
        h[1] += C61*( -5.25*sx*sy*sz*(5. - 60.*sz2 + 99.*sz4) );
        h[2] += C61*( -2.625*sx*(-5. + 100.*sz2 - 285.*sz4 + 198.*sz6) );

// -C_{6,2} dZ_{6,2}/dS

        double C62 = dev_cf_coeffs[22*num_spins + idx];

        h[0] += C62*( -26.25*sx*(2.*sy2 + (19. - 72.*sy2)*sz2 + 6.*(-17. + 33.*sy2)*sz4 + 99.*sz6) );
        h[1] += C62*( -26.25*sy*(-2. + 55.*sz2 - 168.*sz4 + 2.*sy2*(1. - 36.*sz2 + 99.*sz4) + 99.*sz6) );
        h[2] += C62*( -26.25*sz*(-1. + 2.*sy2 + sz2)*(19. - 102.*sz2 + 99.*sz4) );

// -C_{6,3} dZ_{6,3}/dS

        double C63 = dev_cf_coeffs[23*num_spins + idx];

        h[0] += C63*( -472.5*sz*(-1. + 16.*sz2 + 8.*sy4*(-2. + 11.*sz2) - 37.*sz4 + 2.*sy2*(7. - 54.*sz2 + 55.*sz4) + 22.*sz6) );
        h[1] += C63*( 945.*sx*sy*sz*(5. - 24.*sz2 + sy2*(-8. + 44.*sz2) + 11.*sz4) );
        h[2] += C63*( 472.5*sx*(-1. + 4.*sy2 + sz2)*(1. - 15.*sz2 + 22.*sz4) );

// -C_{6,4} dZ_{6,4}/dS

        double C64 = dev_cf_coeffs[24*num_spins + idx];

        h[0] += C64*( 945.*sx*(13.*sz2 + 8.*sy4*(-2. + 33.*sz2) - 46.*sz4 + 8.*sy2*(1. - 24.*sz2 + 33.*sz4) + 33.*sz6) );
        h[1] += C64*( 945.*sy*(-8. + 109.*sz2 + 8.*sy4*(-2. + 33.*sz2) - 134.*sz4 + 8.*sy2*(3. - 46.*sz2 + 33.*sz4) + 33.*sz6) );
        h[2] += C64*( 945.*(sx4 - 6.*sx2*sy2 + sy4)*sz*(-13. + 33.*sz2) );

// -C_{6,5} dZ_{6,5}/dS

        double C65 = dev_cf_coeffs[25*num_spins + idx];

        h[0] += C65*( 10395.*sz*(96.*sy6 + pow(-1. + sz2,2)*(-1. + 6.*sz2) + 8.*sy4*(-16. + 21.*sz2) + 2.*sy2*(19. - 58.*sz2 + 39.*sz4)) );
        h[1] += C65*( -20790.*sx*sy*sz*(13. - 56.*sy2 + 48.*sy4 + 4.*(-4. + 9.*sy2)*sz2 + 3.*sz4) );
        h[2] += C65*( -10395.*sx*(16.*sy4 + 12.*sy2*(-1. + sz2) + pow(-1. + sz2,2))*(-1. + 6.*sz2) );

// -C_{6,6} dZ_{6,6}/dS

        double C66 = dev_cf_coeffs[26*num_spins + idx];

        h[0] += C66*( -62370.*sx*(6.*sy6 + 5.*sy4*sz2 - 10.*sx2*sy2*(2.*sy2 + sz2) + sx4*(6.*sy2 + sz2)) );
        h[1] += C66*( 62370.*sy*(6.*sx6 + sy4*sz2 + 5.*sx4*(-4.*sy2 + sz2) + 2.*sx2*(3.*sy4 - 5.*sy2*sz2)) );
        h[2] += C66*( 62370.*(sx6 - 15.*sx4*sy2 + 15.*sx2*sy4 - 1.*sy6)*sz );

    }
}