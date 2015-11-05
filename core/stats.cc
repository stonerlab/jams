#include <cmath>
#include <numeric>
#include <vector>
#include <iostream>

#include "core/stats.h"
#include "core/maths.h"

double Stats::geweke() {
    int elements;
    double first_10_pc_mean, first_10_pc_var, first_10_pc_sq_sum;
    double last_50_pc_mean, last_50_pc_var, last_50_pc_sq_sum;
    std::vector<double> diff;
    std::vector<double>::const_iterator first_it;
    std::vector<double>::const_iterator last_it;

    // starts for first 10 percent of data
    elements = nint(0.1*data_.size());
    first_it = data_.begin();
    last_it = data_.begin() + elements;

    first_10_pc_mean = std::accumulate(first_it, last_it, 0.0) / double(elements);

    diff.resize(elements);
    std::transform(first_it, last_it, diff.begin(),
                   std::bind2nd(std::minus<double>(), first_10_pc_mean));

    first_10_pc_sq_sum = std::inner_product(diff.begin(), diff.end(), diff.begin(), 0.0);

    first_10_pc_var = (first_10_pc_sq_sum/double(elements-1));


    // starts for first 10 percent of data
    elements = nint(0.5*data_.size());
    first_it = data_.end() - elements;
    last_it = data_.end();

    last_50_pc_mean = std::accumulate(first_it, last_it, 0.0) / double(elements);

    diff.resize(elements);
    std::transform(first_it, last_it, diff.begin(),
                   std::bind2nd(std::minus<double>(), last_50_pc_mean));

    last_50_pc_sq_sum = std::inner_product(diff.begin(), diff.end(), diff.begin(), 0.0);

    last_50_pc_var = (last_50_pc_sq_sum/double(elements-1));

    return (first_10_pc_mean - last_50_pc_mean) / std::sqrt(first_10_pc_var + last_50_pc_var);
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