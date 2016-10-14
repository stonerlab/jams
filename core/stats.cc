#include <cmath>
#include <numeric>
#include <vector>
#include <iostream>
#include <fftw3.h>

#include "core/stats.h"
#include "core/maths.h"

double Stats::spectral_density_zero() {
    return spectral_density_zero(0, data_.size());
}

// Calculates the spectral density about zero, S(0), which can then be used to
// obtain estimates for variance accounting for auto correlation.
//
// The code here is based on:
//     Numerical Methods of Statistics by John F. Monahan
//     http://www4.stat.ncsu.edu/~monahan/jun08/f13/mh1.f95
//
// Notes:
//  - spectral windowing is the same as the above reference (the choice is arbitrary)
//  - x[0] after fft is the mean (i.e. we did not subtract the mean before FFT)

double Stats::spectral_density_zero(const size_t t0, const size_t t1) {
    using std::vector;
    using std::complex;

    assert(t0 >= 0 && t0 < data_.size()-1);
    assert(t1 > 0  && t1 < data_.size());

    fftw_plan       p;                  // fftw planning handle
    int num_samples = t1 - t0;          // number of data samples
    int nd = ceil(sqrt(num_samples));   // size of spectral windowing
    double s0 = 0.0;                    // spectral density at zero (output)

    vector<complex<double>> x(num_samples); // fft result array


    p = fftw_plan_dft_r2c_1d(
        num_samples,                                // size of transform
        &data_[t0],                                 // input
        reinterpret_cast<fftw_complex*>(&x[0]),     // output
        FFTW_ESTIMATE);                             // planner flags

    fftw_execute(p);

    fftw_destroy_plan(p);

    // spectral windowing
    s0 = 0.5 * norm(x[1]);
    for (int k = 0; k < nd; ++k) {
        // skip x[0] because x[0] = mean(x)
        s0 = s0 + norm(x[k + 1]);
    }
    s0 = (2.0 / num_samples) * s0 / double(2 * nd +1);

    return s0;
}

void Stats::geweke(double &diagnostic, double &num_std_err) {
    size_t nN = data_.size();
    size_t nA = 0.1 * nN;
    size_t nB = 0.5 * nN;

    if (nA < 10) {
        diagnostic = std::numeric_limits<double>::infinity();
        num_std_err = std::numeric_limits<double>::infinity();

        return;
    }

    double meanA, meanB, s0A, s0B;

    meanA = mean(0, nA);
    meanB = mean(nB, nN);

    s0A = spectral_density_zero(0, nA);
    s0B = spectral_density_zero(nB, nN);

    // std::cerr << meanA << "\t" << meanB << "\t" << s0A << "\t" << s0B << "\t" << sqrt(s0B/double(nB)) << "\t" << stddev(nB, nN) / sqrt(double(nB)) << std::endl;

    diagnostic = (meanA - meanB) / sqrt(s0A / double(nA) + s0B / double(nB));
    num_std_err =  sqrt(spectral_density_zero(nA, nN) / double(nN - nA));   // based on 90% of data
}

double Stats::median() {
    std::vector<double> sorted_data(data_);
    size_t mid_point = sorted_data.size() / 2;
    std::nth_element(sorted_data.begin(), sorted_data.begin() + mid_point, sorted_data.end());
    return sorted_data[mid_point];
}

double Stats::inter_quartile_range() {
    std::vector<double> sorted_data(data_);
    double upper_quartile, lower_quartile;
    const size_t q = sorted_data.size() / 4;
    const size_t n = sorted_data.size();
    std::sort(sorted_data.begin(), sorted_data.end());

    if (n % 4 == 0) {
        lower_quartile = 0.5 * (sorted_data[q] + sorted_data[q - 1]);
        upper_quartile = 0.5 * (sorted_data[3 * q] + sorted_data[3 * q - 1]);
    } else {
        lower_quartile = sorted_data[q];
        upper_quartile = sorted_data[3 * q];
    }

    return upper_quartile - lower_quartile;
}

void Stats::histogram(std::vector<double> &range, std::vector<double> &bin, double min_value, double max_value, int num_bins) {

    if (data_.size() == 0 && num_bins == 0) {
        bin.resize(1, 0.0);
        range.resize(2, 0.0);
        return;
    }

    if (min_value == max_value) {
        min_value = this->min();
        max_value = this->max();
    }

    // algorithmically choose number of bins
    if (num_bins == 0) {
        // Freedman–Diaconis method
        double bin_size = 2.0 * inter_quartile_range() / cbrt(data_.size());
        if (bin_size < (max_value - min_value)) {
            num_bins = (max_value - min_value) / bin_size;
        } else {
            num_bins = 1;
        }
    }

    bin.resize(num_bins, 0.0);
    range.resize(num_bins + 1, 0.0);


    const double delta = (max_value - min_value) / static_cast<double>(num_bins);

    range[0] = min_value;
    for (int i = 1; i < num_bins + 1; ++i) {
        range[i] = min_value + i * delta;
    }

    for (int i = 0; i < data_.size(); ++i) {
        for (int j = 1; j < num_bins + 1; ++j) {
            if (data_[i] < range[j]) {
                bin[j - 1] += 1;
                break;
            }
        }
    }
}