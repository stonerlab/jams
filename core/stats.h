// Copyright 2015 Joseph Barker. All rights reserved.

#ifndef JAMS_CORE_STATS_H
#define JAMS_CORE_STATS_H

#include <cassert>
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
  inline double min(const size_t t0, const size_t t1);
  inline double max();
  inline double max(const size_t t0, const size_t t1);
  inline double sum();
  inline double sum(const size_t t0, const size_t t1);
  inline double sqsum();
  inline double sqsum(const size_t t0, const size_t t1);
  inline double mean();
  inline double mean(const size_t t0, const size_t t1);
  inline double stddev();
  inline double stddev(const size_t t0, const size_t t1);

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
  return min(0, data_.size());
}

inline double Stats::min(const size_t t0, const size_t t1) {
  assert(t0 >= 0 && t0 < data_.size()-1);
  assert(t1 > 0  && t1 < data_.size());

  if (data_.size() == 0) {
    return 0.0;
  }
  return *std::min_element(data_.begin()+t0, data_.begin()+t1);
}

inline double Stats::max() {
  return max(0, data_.size());
}

inline double Stats::max(const size_t t0, const size_t t1) {
  assert(t0 >= 0 && t0 < data_.size()-1);
  assert(t1 > 0  && t1 < data_.size());

  if (data_.size() == 0) {
    return 0.0;
  }
  return *std::max_element(data_.begin()+t0, data_.begin()+t1);
}

inline void Stats::add(const double &x) {
  data_.push_back(x);
}

inline double Stats::sum() {
  return sum(0, data_.size());
}

inline double Stats::sum(const size_t t0, const size_t t1) {
  assert(t0 >= 0 && t0 < data_.size()-1);
  assert(t1 > 0  && t1 < data_.size());

  return std::accumulate(data_.begin() + t0, data_.begin() + t1, 0.0);
}

inline double Stats::sqsum() {
  return sqsum(0, data_.size());
}


inline double Stats::sqsum(const size_t t0, const size_t t1) {
  assert(t0 >= 0 && t0 < data_.size()-1);
  assert(t1 > 0  && t1 < data_.size());

  return std::inner_product(data_.begin() + t0, data_.begin() + t1, data_.begin() + t0, 0.0);
}

inline double Stats::mean() {
  return mean(0, data_.size());
}

inline double Stats::mean(const size_t t0, const size_t t1) {
  assert(t0 >= 0 && t0 < data_.size()-1);
  assert(t1 > 0  && t1 < data_.size());

  return this->sum(t0, t1)/static_cast<double>(t1 - t0);
}

inline double Stats::stddev() {
  std::vector<double> diff(data_.size());
  std::transform(data_.begin(), data_.end(), diff.begin(),
                 std::bind2nd(std::minus<double>(), this->mean()));
  double sq_sum = std::inner_product(diff.begin(), diff.end(), diff.begin(), 0.0);
  return std::sqrt(sq_sum / data_.size());
}

inline double Stats::stddev(const size_t t0, const size_t t1) {
  std::vector<double> diff(t1 - t0);
  std::transform(data_.begin() + t0, data_.begin() + t1, diff.begin(),
                 std::bind2nd(std::minus<double>(), this->mean(t0, t1)));
  double sq_sum = std::inner_product(diff.begin(), diff.end(), diff.begin(), 0.0);
  return std::sqrt(sq_sum / double(t1 - t0));
}


#endif  // JAMS_CORE_STATS_H
