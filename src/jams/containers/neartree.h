#ifndef JAMS_CORE_NEARTREE_H
#define JAMS_CORE_NEARTREE_H

#include <cassert>
#include <limits>
#include <algorithm>
#include <vector>
#include <random>

#include "jams/helpers/maths.h"

// Toggles whether to do safe floating point comparisons with an 'epslion' or
// use the trivial >=, <= operators
#define SAFE_FLOAT_COMPARISON 1

namespace jams {
    // A class implementing a fast method of finding neighbours in a set of items
    //
    // This is based on an article by Larry Andrews in the C/C++ Users Journal
    // October 2001. The original algorithm is from MacDonald and Kalantari 1983.
    // A different implementation is available at http://neartree.sourceforge.net/
    // which uses loops rather than recursion but is much more complex and appears
    // to have slightly lower performance based on my testing.
    //
    // Norm functor requirements
    // -------------------------
    // The 'norm_functor' MUST obey the triangle inequality, i.e. it must be a
    // proper norm, not simply a distance. The method WILL NOT WORK WITH SQUARED
    // DISTANCES without changes to the code where the triangle rule is used because
    //      || x + y ||^2 <= ||x||^2 + ||y||^2
    //  is not equal to
    //      || x + y ||^2 <= (||x|| + ||y||)^2
    //
    // Floating point comparisons
    // --------------------------
    // These are performed correctly where we check inequalities such as >= and <=
    // by using an 'epsilon' tolerance factor. By default this is set to
    // FLT_EPSLION (i.e. we accept a tolerance for distances of single precision
    // on our double precision values).
    //
    template<typename T, class FuncType, class DistType = double>
    class NearTree {
    public:

        template<typename FType, class FFuncType, class FDistType>
        friend void swap(NearTree<FType, FFuncType, FDistType>& first,
                         NearTree<FType, FFuncType, FDistType>& second);


        NearTree(const NearTree&) = delete;
        NearTree& operator=(const NearTree&) = delete;

        NearTree(NearTree&& other) noexcept
        : left(nullptr), right(nullptr), left_branch(nullptr), right_branch(nullptr),
          max_distance_left(empty_distance()), max_distance_right(empty_distance()), size_(0),
          norm_functor(std::move(other.norm_functor)) {
          swap(*this, other);
        }

        NearTree& operator=(NearTree&& other) noexcept {
          if (this != &other) swap(*this, other);
          return *this;
        }

        explicit NearTree(FuncType func);

        NearTree(FuncType func, const std::vector<T>& items, bool randomize = true);

        ~NearTree();

        void insert(const T &t);

        void insert(std::vector<T> items, bool randomize = true);

        bool nearest_neighbour(const DistType &radius, T &closest, const T &origin) const;

        int num_neighbours_in_radius(const DistType &radius, const T &origin, const DistType &epsilon) const;

        std::vector<T> find_in_radius(const DistType &radius, const T &origin, const DistType &epsilon) const;

        std::vector<T> find_in_annulus(const DistType &inner_radius, const DistType &outer_radius, const T &origin, const DistType &epsilon) const;

        inline std::size_t size() const noexcept { return size_; };

        std::size_t memory() const noexcept;

    private:
        T *left = nullptr;  // left object stored in this node
        T *right = nullptr;  // right object stored in this node

        NearTree *left_branch = nullptr;  // tree descending from the left
        NearTree *right_branch = nullptr;  // tree descending from the right

        DistType max_distance_left = empty_distance();  // longest distance from the left object to anything below it in the tree
        DistType max_distance_right = empty_distance(); // longest distance from the right object to anything below it in the tree
        std::size_t size_ = 0;

        FuncType norm_functor;  // functor to calculate distance

        static constexpr DistType empty_distance() noexcept {
          return std::numeric_limits<DistType>::lowest();
        }

        void
        in_radius(const DistType &radius, std::vector<T> &closest, const T &origin,
                  const DistType &epsilon) const;

        void
        in_annulus(const DistType &inner_radius, const DistType &outer_radius, std::vector<T> &closest,
                   const T &origin, const DistType &epsilon) const;

        bool nearest(DistType &radius, T &closest, const T &origin) const;
    };

    //
    // Constructs an empty NearTree
    //
    template<typename T, typename FuncType, typename DistType>
    NearTree<T, FuncType, DistType>::NearTree(FuncType func)
        : norm_functor(func) {}

    //
    // Constructs a NearTree from a list of items
    //
    // func:      norm functor, WARNING: see class notes for caveats
    // items:     vector of items to be inserted
    // randomize: insert items in random order, generally giving much better
    //               search performance
    //
  template<typename T, typename FuncType, typename DistType>
    NearTree<T, FuncType, DistType>::NearTree(FuncType func, const std::vector<T>& items, bool randomize)
        : norm_functor(func) {
      insert(items, randomize);
    }

    //
    // NearTree destructor
    //
    template<typename T, typename FuncType, typename DistType>
    NearTree<T, FuncType, DistType>::~NearTree() {
      delete left_branch;
      left_branch = nullptr;

      delete right_branch;
      right_branch = nullptr;

      delete left;
      left = nullptr;

      delete right;
      right = nullptr;

      max_distance_left = empty_distance();
      max_distance_right = empty_distance();
      size_ = 0;
    }

    template<typename T, typename FuncType, typename  DistType>
    void NearTree<T, FuncType, DistType>::insert(std::vector<T> items, bool randomize) {
      if (randomize) {
        // Near tree lookups are MUCH more efficient (an order of magnitude)
        // if the inserted positions are randomized, rather than regular in
        // space. Therefore, by default we will randomize the insertions.
        std::random_device rd;
        std::mt19937 rng(rd());
        std::shuffle(items.begin(), items.end(), rng);
      }
      for (auto &x : items) {
        insert(x);
      }
    }

    template<typename T, typename FuncType, typename DistType>
    std::size_t NearTree<T, FuncType, DistType>::memory() const noexcept {
      std::size_t bytes = 0;
      if (left != nullptr) {
        bytes += sizeof(T);
      }
      if (right != nullptr) {
        bytes += sizeof(T);
      }
      if (left_branch != nullptr) {
        bytes += sizeof(NearTree) + left_branch->memory();
      }
      if (right_branch != nullptr) {
        bytes += sizeof(NearTree) + right_branch->memory();
      }
      return bytes;
    }

    //
    // Inserts a single item into the NearTree
    //
    // WARNING: if items are inserted in an ordered manner this can given very
    // poor performance. Preferably construct from a list of items (which will
    // be randomised on insertion).
    //
    template<typename T, typename FuncType, typename DistType>
    void NearTree<T, FuncType, DistType>::insert(const T &t) {
      ++size_;
      // do a bit of precomputing if possible so that we can
      // reduce the number of calls to operator norm_functor as much
      // as possible; norm_functor might use square roots
      DistType tmp_distance_right = (right != nullptr) ? norm_functor(t, *right) : DistType(0);
      DistType tmp_distance_left = (left != nullptr) ? norm_functor(t, *left) : DistType(0);

      if (left == nullptr) {
        left = new T(t);
      } else if (right == nullptr) {
        right = new T(t);
      } else if (tmp_distance_left > tmp_distance_right) {
        if (right_branch == nullptr) {
          right_branch = new NearTree(norm_functor);
        }
        // assumes that max_distance_right is negative for a new node
        max_distance_right = std::max(max_distance_right, tmp_distance_right);
        right_branch->insert(t);
      } else {
        if (left_branch == nullptr) {
          left_branch = new NearTree(norm_functor);
        }
        // assumes that max_distance_left is negative for a new node
        max_distance_left = std::max(max_distance_left, tmp_distance_left);
        left_branch->insert(t);
      }
    }

    //
    // Find the nearest neighbour from the origin with a search radius
    //
    // radius: search radius from origin
    // closest: output nearest item
    // origin: origin from which to search
    //
    // return: true if item is found within search radius
    //
    template<typename T, typename FuncType, typename DistType>
    bool NearTree<T, FuncType, DistType>::nearest_neighbour(const DistType &radius,
                                                  T &closest, const T &origin) const {
      DistType search_radius = radius;
      return (nearest(search_radius, closest, origin));
    }

    //
    // Find all items within this radius (<=) from the chosen origin
    //
    // radius: radius from origin
    // origin: origin from which to search
    //
    // return: vector of the found items
    //
    template<typename T, typename FuncType, typename DistType>
    std::vector<T> NearTree<T, FuncType, DistType>::find_in_radius(const DistType &radius,
                                                         const T &origin, const DistType &epsilon) const {
      std::vector<T> closest;
      in_radius(radius, closest, origin, epsilon);
      return closest;
    }

    //
    // Find all items between inner_radius and outer_radius from the origin
    //
    // inner_radius: inner_radius from origin of annulus
    // outer_radius: outer_radius from origin of annulus
    // origin: origin from which to search
    //
    // return: vector of the found items
    //
    template<typename T, typename FuncType, typename DistType>
    std::vector<T> NearTree<T, FuncType, DistType>::find_in_annulus(const DistType &inner_radius, const DistType &outer_radius,
                                                          const T &origin, const DistType &epsilon) const {
      std::vector<T> closest;
      in_annulus(inner_radius, outer_radius, closest, origin, epsilon);
      return closest;
    }

    template<typename T, typename FuncType, typename DistType>
    bool NearTree<T, FuncType, DistType>::nearest(DistType &radius,
                                        T &closest, const T &origin) const {
      DistType tmp_radius;
      bool is_nearer = false;
      // first test each of the left and right positions to see
      // if one holds a point nearer than the nearest so far.

      const auto norm_left = left != nullptr ? norm_functor(origin, *left) : DistType(0);
      const auto norm_right = right != nullptr ? norm_functor(origin, *right) : DistType(0);

      if ((left != nullptr) &&
          ((tmp_radius = norm_left) <= radius)) {
        radius = tmp_radius;
        closest = *left;
        is_nearer = true;
      }
      if ((right != nullptr) &&
          ((tmp_radius = norm_right) <= radius)) {
        radius = tmp_radius;
        closest = *right;
        is_nearer = true;
      }
      // Now we test to see if the branches below might hold an
      // object nearer than the best so far found. The triangle
      // rule is used to test whether it's even necessary to
      // descend.
      if ((left_branch != nullptr) &&
          ((radius + max_distance_left) >= norm_left)) {
        is_nearer |= left_branch->nearest(radius, closest, origin);
      }

      if ((right_branch != nullptr) &&
          ((radius + max_distance_right) >= norm_right)) {
        is_nearer |= right_branch->nearest(radius, closest, origin);
      }
      return (is_nearer);
    }

    template<typename T, typename FuncType, typename DistType>
    void NearTree<T, FuncType, DistType>::in_radius(const DistType &radius,
                                          std::vector<T> &closest, const T &origin, const DistType &epsilon) const {

      // first test each of the left and right positions to see
      // if one holds a point nearer than the search radius.

      const auto norm_left = left != nullptr ? norm_functor(origin, *left) : DistType(0);
      const auto norm_right = right != nullptr ? norm_functor(origin, *right) : DistType(0);

      #ifdef SAFE_FLOAT_COMPARISON
      if ((left != nullptr) && !definately_greater_than(norm_left, radius, epsilon)) {
        closest.push_back(*left); // It's a keeper
      }
      if ((right != nullptr) && !definately_greater_than(norm_right, radius, epsilon)) {
        closest.push_back(*right); // It's a keeper
      }
      #else
      if ((left != nullptr) && (norm_left <= radius)) {
        closest.push_back(*left); // It's a keeper
      }
      if ((right != nullptr) && (norm_right <= radius)) {
        closest.push_back(*right); // It's a keeper
      }
      #endif
      //
      // Now we test to see if the branches below might hold an
      // object nearer than the search radius. The triangle rule
      // is used to test whether it's even necessary to descend.
      //
      #ifdef SAFE_FLOAT_COMPARISON
      if ((left_branch != nullptr) &&
          !definately_greater_than(norm_left, (radius + max_distance_left), epsilon)) {
        left_branch->in_radius(radius, closest, origin, epsilon);
      }
      if ((right_branch != nullptr) &&
          !definately_greater_than(norm_right, (radius + max_distance_right), epsilon)) {
        right_branch->in_radius(radius, closest, origin, epsilon);
      }
      #else
      if ((left_branch != nullptr) &&
          (radius + max_distance_left) >= norm_left) {
        left_branch->in_radius(radius, closest, origin, epsilon);
      }
      if ((right_branch != nullptr) &&
          (radius + max_distance_right) >= norm_right) {
        right_branch->in_radius(radius, closest, origin, epsilon);
      }
      #endif
    }

    template<typename T, typename FuncType, typename DistType>
    void NearTree<T, FuncType, DistType>::in_annulus(const DistType &inner_radius, const DistType &outer_radius,
                                           std::vector<T> &closest, const T &origin, const DistType& epsilon) const {
      // first test each of the left and right positions to see
      // if one holds a point nearer than the search radius.

      const auto norm_left = left != nullptr ? norm_functor(origin, *left) : DistType(0);
      const auto norm_right = right != nullptr ? norm_functor(origin, *right) : DistType(0);

      if ((left != nullptr)
      && !definately_greater_than(norm_left, outer_radius, epsilon)
        && definately_greater_than(norm_left, inner_radius, epsilon)) {
        closest.push_back(*left); // It's a keeper
      }
      if ((right != nullptr)
          && !definately_greater_than(norm_right, outer_radius, epsilon)
          && definately_greater_than(norm_right, inner_radius, epsilon)) {
        closest.push_back(*right); // It's a keeper
      }
      //
      // Now we test to see if the branches below might hold an
      // object nearer than the search radius. The triangle rule
      // is used to test whether it's even necessary to descend.
      //

      if ((left_branch != nullptr) &&
          !definately_greater_than(norm_left, (outer_radius + max_distance_left), epsilon)) {
        left_branch->in_annulus(inner_radius, outer_radius, closest, origin, epsilon);
      }
      if ((right_branch != nullptr) &&
          !definately_greater_than(norm_right, (outer_radius + max_distance_right), epsilon)) {
        right_branch->in_annulus(inner_radius, outer_radius, closest, origin, epsilon);
      }
    }

    template<typename T, typename FuncType, typename DistType>
    int NearTree<T, FuncType, DistType>::num_neighbours_in_radius(const DistType &radius, const T &origin,
                                                        const DistType &epsilon) const {
      int num_neighbours = 0;
      // first test each of the left and right positions to see
      // if one holds a point nearer than the search radius.

      #ifdef SAFE_FLOAT_COMPARISON
      if ((left != nullptr) && !definately_greater_than(norm_functor(origin, *left), radius, epsilon)) {
        num_neighbours++;
      }
      if ((right != nullptr) && !definately_greater_than(norm_functor(origin, *right), radius, epsilon)) {
        num_neighbours++;
      }
      #else
      if ((left != nullptr) && (norm_functor(origin, *left) <= radius)) {
        num_neighbours++;
      }
      if ((right != nullptr) && (norm_functor(origin, *right) <= radius)) {
        num_neighbours++;
      }
      #endif
      //
      // Now we test to see if the branches below might hold an
      // object nearer than the search radius. The triangle rule
      // is used to test whether it's even necessary to descend.
      //
      #ifdef SAFE_FLOAT_COMPARISON
      if ((left_branch != nullptr) &&
        !definately_greater_than(norm_functor(origin, *left), (radius + max_distance_left), epsilon)) {
        num_neighbours += left_branch->num_neighbours_in_radius(radius, origin, epsilon);
      }
      if ((right_branch != nullptr) &&
        !definately_greater_than(norm_functor(origin, *right), (radius + max_distance_right), epsilon)) {
        num_neighbours += right_branch->num_neighbours_in_radius(radius, origin, epsilon);
      }
      #else
      if ((left_branch != nullptr) &&
          (radius + max_distance_left) >= norm_functor(origin, *left)) {
        num_neighbours += left_branch->num_neighbours_in_radius(radius, origin, epsilon);
      }
      if ((right_branch != nullptr) &&
          (radius + max_distance_right) >= norm_functor(origin, *right)) {
        num_neighbours += right_branch->num_neighbours_in_radius(radius, origin, epsilon);
      }
      #endif
      return num_neighbours;
    }

    template<typename T, typename FuncType, typename DistType>
    void swap(NearTree<T, FuncType, DistType>& first, NearTree<T, FuncType, DistType>& second) {
      using std::swap;

      swap(first.left, second.left);
      swap(first.right, second.right);
      swap(first.left_branch, second.left_branch);
      swap(first.right_branch, second.right_branch);
      swap(first.max_distance_left, second.max_distance_left);
      swap(first.max_distance_right, second.max_distance_right);
      swap(first.size_, second.size_);
      swap(first.norm_functor, second.norm_functor);
    }
}

#endif  // JAMS_CORE_NEARTREE_H
