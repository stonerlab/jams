// Copyright 2015 Joseph Barker. All rights reserved.

#ifndef JAMS_CORE_STATS_H
#define JAMS_CORE_STATS_H

#include <algorithm>
#include <numeric>
#include <vector>
#include <functional>

class Stats {
 public:
  Stats() : data_() {};

  // copy constructor
  Stats(const Stats& other)
        : data_(other.data_) {};

  // construct from vector
  Stats(const std::vector<double>& vec)
        : data_(vec) {};

  ~Stats() {};


  friend void swap(Stats& first, Stats& second) // nothrow
  {
      using std::swap;
      swap(first.data_, second.data_);
  }

  Stats& operator=(Stats other) {
      swap(*this, other);
      return *this;
  }

  inline void add(const double &x);
  inline size_t size();

  inline double min();
  inline double max();
  inline double sum();
  inline double sqsum();
  inline double mean();
  inline double stddev();
  double median();
  double inter_quartile_range();

  double geweke();
  void histogram(std::vector<double> &range, std::vector<double> &bin, double min_value = 0, double max_value = 0, int num_bins = 0);

 protected:
  std::vector<double> data_;
};

inline size_t Stats::size() {
  return data_.size();
}

inline double Stats::min() {
  if (data_.size() == 0) {
    return 0.0;
  }
  return *std::min_element(data_.begin(), data_.end());
}

inline double Stats::max() {
  if (data_.size() == 0) {
    return 0.0;
  }
  return *std::max_element(data_.begin(), data_.end());
}

inline void Stats::add(const double &x) {
  data_.push_back(x);
}

inline double Stats::sum() {
  return std::accumulate(data_.begin(), data_.end(), 0.0);
}

inline double Stats::sqsum() {
  return std::inner_product(data_.begin(), data_.end(), data_.begin(), 0.0);
}

inline double Stats::mean() {
  return this->sum()/static_cast<double>(data_.size());
}

inline double Stats::stddev() {
  std::vector<double> diff(data_.size());
  std::transform(data_.begin(), data_.end(), diff.begin(),
                 std::bind2nd(std::minus<double>(), this->mean()));
  double sq_sum = std::inner_product(diff.begin(), diff.end(), diff.begin(), 0.0);
  return std::sqrt(sq_sum / data_.size());
}


#endif  // JAMS_CORE_STATS_H
