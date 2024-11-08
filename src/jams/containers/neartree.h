#ifndef JAMS_CORE_NEARTREE_H
#define JAMS_CORE_NEARTREE_H

#include <limits>
#include <stack>
#include <cfloat>
#include <algorithm>
#include <vector>
#include <random>

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
    template<typename T, class FuncType>
    class NearTree {
    public:

        template<typename FType, class FFuncType>
        friend void swap(NearTree<FType, FFuncType>& first, NearTree<FType, FFuncType>& second);

        NearTree& operator=(NearTree other) {
          swap(*this, other);
          return *this;
        }

        NearTree(NearTree&& other) noexcept
        : NearTree(other.norm_functor) {
          swap(*this, other);
        }

        explicit NearTree(FuncType func);

        NearTree(FuncType func, const std::vector<T>& items, bool randomize = true);

        ~NearTree();

        void insert(const T &t);

        void insert(std::vector<T> items, bool randomize = true);

        bool nearest_neighbour(const double &radius, T &closest, const T &origin) const;

        int num_neighbours_in_radius(const double &radius, const T &origin, const double &epsilon) const;

        std::vector<T> find_in_radius(const double &radius, const T &origin, const double &epsilon) const;

        std::vector<T> find_in_annulus(const double &inner_radius, const double &outer_radius, const T &origin, const double &epsilon) const;

        inline std::size_t size() const { return node_count(); };

        inline std::size_t memory() const {
          return node_count() * (2 * sizeof(T *) + 2 * sizeof(NearTree *) + 2 * sizeof(double));
        };

    private:
        T *left = nullptr;  // left object stored in this node
        T *right = nullptr;  // right object stored in this node

        NearTree *left_branch = nullptr;  // tree descending from the left
        NearTree *right_branch = nullptr;  // tree descending from the right

        double max_distance_left = -1;  // longest distance from the left object to anything below it in the tree
        double max_distance_right = -1; // longest distance from the right object to anything below it in the tree

        FuncType norm_functor;  // functor to calculate distance

        void
        in_radius(const double &radius, std::vector<T> &closest, const T &origin,
                  const double &epsilon) const;

        void
        in_annulus(const double &inner_radius, const double &outer_radius, std::vector<T> &closest,
                   const T &origin, const double &epsilon) const;

        bool nearest(double &radius, T &closest, const T &origin) const;

        std::size_t node_count() const;
    };

    //
    // Constructs an empty NearTree
    //
    template<typename T,
        typename FuncType>
    NearTree<T, FuncType>::NearTree(FuncType func)
        : norm_functor(func) {}

    //
    // Constructs a NearTree from a list of items
    //
    // func:      norm functor, WARNING: see class notes for caveats
    // items:     vector of items to be inserted
    // randomize: insert items in random order, generally giving much better
    //               search performance
    //
    template<typename T,
        typename FuncType>
    NearTree<T, FuncType>::NearTree(FuncType func, const std::vector<T>& items, bool randomize)
        : norm_functor(func) {
      insert(items, randomize);
    }

    //
    // NearTree destructor
    //
    template<typename T,
        typename FuncType>
    NearTree<T, FuncType>::~NearTree() {
      delete left_branch;
      left_branch = nullptr;

      delete right_branch;
      right_branch = nullptr;

      delete left;
      left = nullptr;

      delete right;
      right = nullptr;

      max_distance_left = -1;
      max_distance_right = -1;
    }

    template<typename T, typename FuncType>
    void NearTree<T, FuncType>::insert(std::vector<T> items, bool randomize) {
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

    //
    // Inserts a single item into the NearTree
    //
    // WARNING: if items are inserted in an ordered manner this can given very
    // poor performance. Preferably construct from a list of items (which will
    // be randomised on insertion).
    //
    template<typename T,
        typename FuncType>
    void NearTree<T, FuncType>::insert(const T &t) {
      // do a bit of precomputing if possible so that we can
      // reduce the number of calls to operator norm_functor as much
      // as possible; norm_functor might use square roots
      double tmp_distance_right = (right != nullptr) ? norm_functor(t, *right) : 0.0;
      double tmp_distance_left = (left != nullptr) ? norm_functor(t, *left) : 0.0;

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
    template<typename T,
        typename FuncType>
    bool NearTree<T, FuncType>::nearest_neighbour(const double &radius,
                                                  T &closest, const T &origin) const {
      double search_radius = radius;
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
    template<typename T,
        typename FuncType>
    std::vector<T> NearTree<T, FuncType>::find_in_radius(const double &radius,
                                                         const T &origin, const double &epsilon) const {
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
    template<typename T,
        typename FuncType>
    std::vector<T> NearTree<T, FuncType>::find_in_annulus(const double &inner_radius, const double &outer_radius,
                                                          const T &origin, const double &epsilon) const {
      std::vector<T> closest;
      in_annulus(inner_radius, outer_radius, closest, origin, epsilon);
      return closest;
    }

    template<typename T,
        typename FuncType>
    bool NearTree<T, FuncType>::nearest(double &radius,
                                        T &closest, const T &origin) const {
      double tmp_radius;
      bool is_nearer = false;
      // first test each of the left and right positions to see
      // if one holds a point nearer than the nearest so far.
      if ((left != nullptr) &&
          ((tmp_radius = (norm_functor(origin, *left))) <= radius)) {
        radius = tmp_radius;
        closest = *left;
        is_nearer = true;
      }
      if ((right != nullptr) &&
          ((tmp_radius = (norm_functor(origin, *right))) <= radius)) {
        radius = tmp_radius;
        closest = *right;
        is_nearer = true;
      }
      // Now we test to see if the branches below might hold an
      // object nearer than the best so far found. The triangle
      // rule is used to test whether it's even necessary to
      // descend.
      if ((left_branch != nullptr) &&
          ((radius + max_distance_left) >= (norm_functor(origin, *left)))) {
        is_nearer |= left_branch->nearest(radius, closest, origin);
      }

      if ((right_branch != nullptr) &&
          ((radius + max_distance_right) >= (norm_functor(origin, *right)))) {
        is_nearer |= right_branch->nearest(radius, closest, origin);
      }
      return (is_nearer);
    }

    template<typename T,
        typename FuncType>
    void NearTree<T, FuncType>::in_radius(const double &radius,
                                          std::vector<T> &closest, const T &origin, const double &epsilon) const {

      // first test each of the left and right positions to see
      // if one holds a point nearer than the search radius.

      #ifdef SAFE_FLOAT_COMPARISON
      if ((left != nullptr) && !definately_greater_than(norm_functor(origin, *left), radius, epsilon)) {
        closest.push_back(*left); // It's a keeper
      }
      if ((right != nullptr) && !definately_greater_than(norm_functor(origin, *right), radius, epsilon)) {
        closest.push_back(*right); // It's a keeper
      }
      #else
      if ((left != nullptr) && (norm_functor(origin, *left) <= radius)) {
        closest.push_back(*left); // It's a keeper
      }
      if ((right != nullptr) && (norm_functor(origin, *right) <= radius)) {
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
          !definately_greater_than(norm_functor(origin, *left), (radius + max_distance_left), epsilon)) {
        left_branch->in_radius(radius, closest, origin, epsilon);
      }
      if ((right_branch != nullptr) &&
          !definately_greater_than(norm_functor(origin, *right), (radius + max_distance_right), epsilon)) {
        right_branch->in_radius(radius, closest, origin, epsilon);
      }
      #else
      if ((left_branch != nullptr) &&
          (radius + max_distance_left) >= norm_functor(origin, *left)) {
        left_branch->in_radius(radius, closest, origin);
      }
      if ((right_branch != nullptr) &&
          (radius + max_distance_right) >= norm_functor(origin, *right)) {
        right_branch->in_radius(radius, closest, origin);
      }
      #endif
    }

    template<typename T,
        typename FuncType>
    void NearTree<T, FuncType>::in_annulus(const double &inner_radius, const double &outer_radius,
                                           std::vector<T> &closest, const T &origin, const double& epsilon) const {
      // first test each of the left and right positions to see
      // if one holds a point nearer than the search radius.

      if ((left != nullptr)
      && !definately_greater_than(norm_functor(origin, *left), outer_radius, epsilon)
        && definately_greater_than(norm_functor(origin, *left), inner_radius, epsilon)) {
        closest.push_back(*left); // It's a keeper
      }
      if ((right != nullptr)
          && !definately_greater_than(norm_functor(origin, *right), outer_radius, epsilon)
          && definately_greater_than(norm_functor(origin, *right), inner_radius, epsilon)) {
        closest.push_back(*right); // It's a keeper
      }
      //
      // Now we test to see if the branches below might hold an
      // object nearer than the search radius. The triangle rule
      // is used to test whether it's even necessary to descend.
      //

      if ((left_branch != nullptr) &&
          !definately_greater_than(norm_functor(origin, *left), (outer_radius + max_distance_left), epsilon)) {
        left_branch->in_annulus(inner_radius, outer_radius, closest, origin, epsilon);
      }
      if ((right_branch != nullptr) &&
          !definately_greater_than(norm_functor(origin, *right), (outer_radius + max_distance_right), epsilon)) {
        right_branch->in_annulus(inner_radius, outer_radius, closest, origin, epsilon);
      }
    }

    template<typename T,
        typename FuncType>
    std::size_t NearTree<T, FuncType>::node_count() const {
      std::size_t count = 0;
      if ((left != nullptr)) {
        count++;
      }
      if ((right != nullptr)) {
        count++;
      }

      if ((left_branch != nullptr)) {
        count += left_branch->node_count();
      }
      if ((right_branch != nullptr)) {
        count += right_branch->node_count();
      }
      return count;
    }

    template<typename T, class FuncType>
    int NearTree<T, FuncType>::num_neighbours_in_radius(const double &radius, const T &origin,
                                                        const double &epsilon) const {
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
        num_neighbours += left_branch->num_neighbours_in_radius(radius, origin);
      }
      if ((right_branch != nullptr) &&
          (radius + max_distance_right) >= norm_functor(origin, *right)) {
        num_neighbours += right_branch->num_neighbours_in_radius(radius, origin);
      }
      #endif
      return num_neighbours;
    }

    template<typename T, class FuncType>
    void swap(NearTree<T, FuncType>& first, NearTree<T, FuncType>& second) {
      using std::swap;

      swap(first.left, second.left);
      swap(first.right, second.right);
      swap(first.left_branch, second.left_branch);
      swap(first.max_distance_left, second.max_distance_left);
      swap(first.max_distance_right, second.max_distance_right);
      swap(first.norm_functor, second.norm_functor);
    }
}

#endif  // JAMS_CORE_LATTICE_H
