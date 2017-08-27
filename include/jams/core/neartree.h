#ifndef JAMS_CORE_NEARTREE_H
#define JAMS_CORE_NEARTREE_H

#include <stack>
#include <cfloat>
#include <algorithm>

// based on http://www.drdobbs.com/cpp/a-template-for-the-nearest-neighbor-prob/184401449
// also library seems to be available at: http://neartree.sourceforge.net/
//==============================================================
template <typename T,
          typename FuncType>
class NearTree
{
    T * left;                  // left object stored in this node
    T * right;                 // right object stored in this node
    double max_distance_left;  // longest distance from the left object
                               // to anything below it in the tree
    double max_distance_right; // longest distance from the right object
                               // to anything below it in the tree
    NearTree * left_branch;    // tree descending from the left
    NearTree * right_branch;   // tree descending from the right

    FuncType distance_functor; // functor to calculate distance

public:

//==============================================================

   NearTree(FuncType func)
    : left(0),
      right(0),
      max_distance_left(DBL_MIN),
      max_distance_right(DBL_MIN),
      left_branch(0),
      right_branch(0),
      distance_functor(func) // constructor
   {}  //  NearTree constructor

   NearTree(FuncType func, std::vector<T> items, bool randomize = true)
    : left(0),
      right(0),
      max_distance_left(DBL_MIN),
      max_distance_right(DBL_MIN),
      left_branch(0),
      right_branch(0),
      distance_functor(func) // constructor
   {
      // Near tree lookups are MUCH more efficient (an order of magnitude)
      // if the inserted positions are randomized, rather than regular in
      // space. Therefore, by default we will randomize the insertions from
      // a vector constructor.
      if (randomize) {
        std::random_shuffle(items.begin(), items.end());
      }
      for (auto it = items.begin(); it != items.end(); ++it) {
        insert(*it);
      }
   }  //  NearTree constructor

//==============================================================
   ~NearTree(void)  // destructor
   {
      delete left_branch;
      left_branch = 0;

      delete right_branch;
      right_branch = 0;

      delete left;
      left = 0;

      delete right;
      right = 0;

      max_distance_left     = DBL_MIN;
      max_distance_right    = DBL_MIN;
   }  //  ~NearTree

//==============================================================
   void insert(const T& t)
   {
      // do a bit of precomputing if possible so that we can
      // reduce the number of calls to operator 'double' as much
      // as possible; 'double' might use square roots
      double tmp_distance_right =  0;
      double tmp_distance_left  =  0;
      if (right != 0) {
         tmp_distance_right  = distance_functor(t, *right);
         tmp_distance_left   = distance_functor(t, *left);
      }
      if (left == 0) {
         left = new T(t);
      }
      else if (right == 0) {
         right   = new T(t);
      }
      else if (tmp_distance_left > tmp_distance_right) {
         if (right_branch == 0) {
           right_branch = new NearTree(distance_functor);
         }
         // note that the next line assumes that max_distance_right
         // is negative for a new node
         if (max_distance_right < tmp_distance_right) {
           max_distance_right = tmp_distance_right;
         }
         right_branch->insert(t);
      }
      else {
         if (left_branch == 0) {
           left_branch = new NearTree(distance_functor);
         }
         // note that the next line assumes that max_distance_left
         // is negative for a new node
         if (max_distance_left < tmp_distance_left) {
           max_distance_left = tmp_distance_left;
         }
         left_branch->insert(t);
      }
   }  //  insert

//==============================================================
   bool nearest_neighbour ( const double& radius,
                             T& closest,   const T& t ) const
   {
      double search_radius = radius;
      return ( nearest ( search_radius, closest, t ) );
   }  //  nearest_neighbour

//==============================================================
   bool farthest_neighbour ( T& farthest, const T& t ) const
   {
      double search_radius = DBL_MIN;
      return ( find_farthest (search_radius, farthest, t));
   }  //  farthest_neighbour

//==============================================================
   void find_in_radius ( const double& radius,
               std::vector<T>& closest,   const T& t ) const
   {   // t is the probe point, closest is a vector of contacts
      // clear the contents of the return vector so that
      // things don't accidentally accumulate
      closest.clear( );
      in_radius( radius, closest, t );
   }  //  find_in_radius

private:

//==============================================================
   bool nearest ( double& radius,
                       T& closest,   const T& t ) const
   {
      double   tmp_radius;
      bool  is_nearer = false;
      // first test each of the left and right positions to see
      // if one holds a point nearer than the nearest so far.
      if (( left != 0 ) &&
        ((tmp_radius = (distance_functor(t, *left))) <= radius))
      {
         radius  = tmp_radius;
         closest = *left;
         is_nearer     = true;
      }
      if ((right != 0) &&
        (( tmp_radius = (distance_functor(t, *right)))<=radius))
      {
         radius  = tmp_radius;
         closest = *right;
         is_nearer     = true;
      }
      // Now we test to see if the branches below might hold an
      // object nearer than the best so far found. The triangle
      // rule is used to test whether it's even necessary to
      // descend.
      if (( left_branch  != 0 )  &&
        ((radius + max_distance_left) >= (distance_functor(t, *left))))
      {
         is_nearer |= left_branch->nearest(radius,closest,t);
      }

      if (( right_branch != 0 )  &&
        ((radius + max_distance_right) >= (distance_functor(t, *right))))
      {
         is_nearer |= right_branch->nearest(radius,closest,t);
      }
      return ( is_nearer );
   };   // nearest

//==============================================================
   void in_radius ( const double& radius,
               std::vector<T>& closest,   const T& t ) const
   {
      // t is the probe point, closest is a vector of contacts

       // first test each of the left and right positions to see
       // if one holds a point nearer than the search radius.

      if ((left != 0) && (distance_functor(t, *left) <= radius)) {
         closest.push_back( *left ); // It's a keeper
      }
      if ((right != 0) && (distance_functor(t, *right) <= radius)) {
         closest.push_back( *right ); // It's a keeper
      }
      //
      // Now we test to see if the branches below might hold an
      // object nearer than the search radius. The triangle rule
      // is used to test whether it's even necessary to descend.
      //
      if (( left_branch  != 0 )  &&
         (radius + max_distance_left) >= distance_functor(t, *left)) {
         left_branch->in_radius( radius, closest, t );
      }
      if (( right_branch != 0 )  &&
         (radius + max_distance_right) >= distance_functor(t, *right)) {
         right_branch->in_radius( radius, closest, t );
      }
   }  //  in_radius
};

#endif  // JAMS_CORE_LATTICE_H
