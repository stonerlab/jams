#ifndef JAMS_CORE_NEARTREE_H
#define JAMS_CORE_NEARTREE_H

#include <stack>
#include <cfloat>
#include <algorithm>

// based on http://www.drdobbs.com/cpp/a-template-for-the-nearest-neighbor-prob/184401449
// also library seems to be available at: http://neartree.sourceforge.net/

template<typename T,
        class FuncType>
class NearTree {
public:

    explicit NearTree(FuncType func);

    NearTree(FuncType func, std::vector<T> items, bool randomize = true);

    ~NearTree();

    void insert(const T &t);

    bool nearest_neighbour(const double &radius, T &closest, const T &t) const;
    void find_in_radius(const double &radius, std::vector<T> &closest, const T &t) const;

private:
    T *left = nullptr;                  // left object stored in this node
    T *right = nullptr;                  // right object stored in this node

    NearTree *left_branch = nullptr;    // tree descending from the left
    NearTree *right_branch = nullptr;    // tree descending from the right

    FuncType distance_functor; // functor to calculate distance

    double max_distance_left = DBL_MIN;  // longest distance from the left object to anything below it in the tree
    double max_distance_right = DBL_MIN; // longest distance from the right object to anything below it in the tree

    void in_radius(const double &radius, std::vector<T> &closest, const T &t) const;
    bool nearest(double &radius, T &closest, const T &t) const;
};

template<typename T,
        typename FuncType>
NearTree<T,FuncType>::NearTree(FuncType func)
        : distance_functor(func) {}

template<typename T,
        typename FuncType>
NearTree<T,FuncType>::NearTree(FuncType func, std::vector<T> items, bool randomize)
        : distance_functor(func) {
  // Near tree lookups are MUCH more efficient (an order of magnitude)
  // if the inserted positions are randomized, rather than regular in
  // space. Therefore, by default we will randomize the insertions from
  // a vector constructor.
  if (randomize) {
    std::random_shuffle(items.begin(), items.end());
  }
  for (auto &x : items) {
    insert(x);
  }
}

template<typename T,
        typename FuncType>
NearTree<T,FuncType>::~NearTree() {
  delete left_branch;
  left_branch = nullptr;

  delete right_branch;
  right_branch = nullptr;

  delete left;
  left = nullptr;

  delete right;
  right = nullptr;

  max_distance_left = DBL_MIN;
  max_distance_right = DBL_MIN;
}

template<typename T,
        typename FuncType>
void NearTree<T,FuncType>::insert(const T &t) {
  // do a bit of precomputing if possible so that we can
  // reduce the number of calls to operator 'double' as much
  // as possible; 'double' might use square roots
  double tmp_distance_right = 0;
  double tmp_distance_left = 0;
  if (right != nullptr) {
    tmp_distance_right = distance_functor(t, *right);
    tmp_distance_left = distance_functor(t, *left);
  }
  if (left == nullptr) {
    left = new T(t);
  } else if (right == nullptr) {
    right = new T(t);
  } else if (tmp_distance_left > tmp_distance_right) {
    if (right_branch == nullptr) {
      right_branch = new NearTree(distance_functor);
    }
    // note that the next line assumes that max_distance_right
    // is negative for a new node
    if (max_distance_right < tmp_distance_right) {
      max_distance_right = tmp_distance_right;
    }
    right_branch->insert(t);
  } else {
    if (left_branch == nullptr) {
      left_branch = new NearTree(distance_functor);
    }
    // note that the next line assumes that max_distance_left
    // is negative for a new node
    if (max_distance_left < tmp_distance_left) {
      max_distance_left = tmp_distance_left;
    }
    left_branch->insert(t);
  }
}

template<typename T,
        typename FuncType>
bool NearTree<T,FuncType>::nearest_neighbour(const double &radius,
                                 T &closest, const T &t) const {
  double search_radius = radius;
  return (nearest(search_radius, closest, t));
}  //  nearest_neighbour

template<typename T,
        typename FuncType>
void NearTree<T,FuncType>::find_in_radius(const double &radius,
                              std::vector<T> &closest,
                              const T &t) const {   // t is the probe point, closest is a vector of contacts
  // clear the contents of the return vector so that
  // things don't accidentally accumulate
  closest.clear();
  in_radius(radius, closest, t);
}

template<typename T,
        typename FuncType>
bool NearTree<T,FuncType>::nearest(double &radius,
                       T &closest, const T &t) const {
  double tmp_radius;
  bool is_nearer = false;
  // first test each of the left and right positions to see
  // if one holds a point nearer than the nearest so far.
  if ((left != nullptr) &&
      ((tmp_radius = (distance_functor(t, *left))) <= radius)) {
    radius = tmp_radius;
    closest = *left;
    is_nearer = true;
  }
  if ((right != nullptr) &&
      ((tmp_radius = (distance_functor(t, *right))) <= radius)) {
    radius = tmp_radius;
    closest = *right;
    is_nearer = true;
  }
  // Now we test to see if the branches below might hold an
  // object nearer than the best so far found. The triangle
  // rule is used to test whether it's even necessary to
  // descend.
  if ((left_branch != nullptr) &&
      ((radius + max_distance_left) >= (distance_functor(t, *left)))) {
    is_nearer |= left_branch->nearest(radius, closest, t);
  }

  if ((right_branch != nullptr) &&
      ((radius + max_distance_right) >= (distance_functor(t, *right)))) {
    is_nearer |= right_branch->nearest(radius, closest, t);
  }
  return (is_nearer);
}

template<typename T,
        typename FuncType>
void NearTree<T,FuncType>::in_radius(const double &radius,
                         std::vector<T> &closest, const T &t) const {
  // t is the probe point, closest is a vector of contacts

  // first test each of the left and right positions to see
  // if one holds a point nearer than the search radius.

  if ((left != nullptr) && (distance_functor(t, *left) <= radius)) {
    closest.push_back(*left); // It's a keeper
  }
  if ((right != nullptr) && (distance_functor(t, *right) <= radius)) {
    closest.push_back(*right); // It's a keeper
  }
  //
  // Now we test to see if the branches below might hold an
  // object nearer than the search radius. The triangle rule
  // is used to test whether it's even necessary to descend.
  //
  if ((left_branch != nullptr) &&
      (radius + max_distance_left) >= distance_functor(t, *left)) {
    left_branch->in_radius(radius, closest, t);
  }
  if ((right_branch != nullptr) &&
      (radius + max_distance_right) >= distance_functor(t, *right)) {
    right_branch->in_radius(radius, closest, t);
  }
}

#endif  // JAMS_CORE_LATTICE_H
