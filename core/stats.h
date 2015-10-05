// Copyright 2015 Joseph Barker. All rights reserved.

#ifndef JAMS_CORE_STATS_H
#define JAMS_CORE_STATS_H

#include <algorithm>
#include <numeric>
#include <vector>

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

  inline double sum();
  inline double sqsum();
  inline double mean();
  inline double stddev();
  double geweke();

 protected:
  std::vector<double> data_;
};

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
  return this->sum()/data_.size();
}

inline double Stats::stddev() {
  std::vector<double> diff(data_.size());
  std::transform(data_.begin(), data_.end(), diff.begin(),
                 std::bind2nd(std::minus<double>(), this->mean()));
  double sq_sum = std::inner_product(diff.begin(), diff.end(), diff.begin(), 0.0);
  return std::sqrt(sq_sum / data_.size());
}


#endif  // JAMS_CORE_STATS_H
