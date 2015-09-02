#include <cmath>
#include <numeric>
#include <vector>

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