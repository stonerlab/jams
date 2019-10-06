//
// Created by Joseph Barker on 2019-04-24.
//

#ifndef JAMS_SORT_H
#define JAMS_SORT_H

// Finding a sort permutation
// https://stackoverflow.com/questions/17074324/how-can-i-sort-two-vectors-in-the-same-way-with-criteria-that-uses-only-one-of

#include <numeric>

template <class T>
std::vector<std::size_t> sort_permutation(const std::vector<T>& vec) {
  std::vector<std::size_t> p(vec.size());
  std::iota(p.begin(), p.end(), 0);
  std::sort(p.begin(), p.end(),
            [&](std::size_t i, std::size_t j){ return vec[i] < vec[j]; });
  return p;
}

template <class T, class Compare>
std::vector<std::size_t> sort_permutation(const jams::MultiArray<T,1>& vec) {
  std::vector<std::size_t> p(vec.size());
  std::iota(p.begin(), p.end(), 0);
  std::sort(p.begin(), p.end(),
            [&](std::size_t i, std::size_t j){ return vec(i) < vec(j); });
  return p;
}

template <class T, class Compare>
std::vector<std::size_t> sort_permutation(
    const std::vector<T>& vec,
    Compare& compare = std::less<T>())
{
  std::vector<std::size_t> p(vec.size());
  std::iota(p.begin(), p.end(), 0);
  std::sort(p.begin(), p.end(),
            [&](std::size_t i, std::size_t j){ return compare(vec[i], vec[j]); });
  return p;
}

template <class T, class Compare>
std::vector<std::size_t> sort_permutation(
    const jams::MultiArray<T,1>& vec,
    Compare& compare = std::less<T>())
{
  std::vector<std::size_t> p(vec.size());
  std::iota(p.begin(), p.end(), 0);
  std::sort(p.begin(), p.end(),
            [&](std::size_t i, std::size_t j){ return compare(vec(i), vec(j)); });
  return p;
}

template <class T>
std::vector<std::size_t> stable_sort_permutation(const std::vector<T>& vec) {
  std::vector<std::size_t> p(vec.size());
  std::iota(p.begin(), p.end(), 0);
  std::stable_sort(p.begin(), p.end(),
                   [&](std::size_t i, std::size_t j){ return vec[i] < vec[j]; });
  return p;
}

template <class T>
std::vector<std::size_t> stable_sort_permutation(const jams::MultiArray<T,1>& vec) {
  std::vector<std::size_t> p(vec.size());
  std::iota(p.begin(), p.end(), 0);
  std::stable_sort(p.begin(), p.end(),
                   [&](std::size_t i, std::size_t j){ return vec(i) < vec(j); });
  return p;
}

template <class T, class Compare>
std::vector<std::size_t> stable_sort_permutation(
    const std::vector<T>& vec,
    Compare& compare = std::less<T>())
{
  std::vector<std::size_t> p(vec.size());
  std::iota(p.begin(), p.end(), 0);
  std::stable_sort(p.begin(), p.end(),
            [&](std::size_t i, std::size_t j){ return compare(vec[i], vec[j]); });
  return p;
}

template <class T, class Compare>
std::vector<std::size_t> stable_sort_permutation(
    const jams::MultiArray<T,1>& vec,
    Compare& compare = std::less<T>())
{
  std::vector<std::size_t> p(vec.size());
  std::iota(p.begin(), p.end(), 0);
  std::stable_sort(p.begin(), p.end(),
                   [&](std::size_t i, std::size_t j){ return compare(vec(i), vec(j)); });
  return p;
}

template <typename T>
std::vector<T> apply_permutation(
    const std::vector<T>& vec,
    const std::vector<std::size_t>& p)
{
  std::vector<T> sorted_vec(vec.size());
  std::transform(p.begin(), p.end(), sorted_vec.begin(),
                 [&](std::size_t i){ return vec[i]; });
  return sorted_vec;
}

template <typename T>
jams::MultiArray<T,1> apply_permutation(
    const jams::MultiArray<T,1>& vec,
    const std::vector<std::size_t>& p)
{
  std::vector<T> sorted_vec(vec.size());
  std::transform(p.begin(), p.end(), sorted_vec.begin(),
                 [&](std::size_t i){ return vec(i); });
  return sorted_vec;
}

template <typename T>
void apply_permutation_in_place(
    std::vector<T>& vec,
    const std::vector<std::size_t>& p)
{
  std::vector<bool> done(vec.size());
  for (std::size_t i = 0; i < vec.size(); ++i)
  {
    if (done[i])
    {
      continue;
    }
    done[i] = true;
    std::size_t prev_j = i;
    std::size_t j = p[i];
    while (i != j)
    {
      std::swap(vec[prev_j], vec[j]);
      done[j] = true;
      prev_j = j;
      j = p[j];
    }
  }
}

template <typename T>
void apply_permutation_in_place(
    jams::MultiArray<T,1>& vec,
    const std::vector<std::size_t>& p)
{
  std::vector<bool> done(vec.size());
  for (std::size_t i = 0; i < vec.size(); ++i)
  {
    if (done[i])
    {
      continue;
    }
    done[i] = true;
    std::size_t prev_j = i;
    std::size_t j = p[i];
    while (i != j)
    {
      std::swap(vec(prev_j), vec(j));
      done[j] = true;
      prev_j = j;
      j = p[j];
    }
  }
}

#endif //JAMS_SORT_H
