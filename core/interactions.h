#ifndef JAMS_CORE_INTERACTIONS_H
#define JAMS_CORE_INTERACTIONS_H

#include <string>
#include <vector>
#include <map>

#include "core/types.h"

template <class T>
class InteractionList;

typedef struct {
    int index;
    jblib::Matrix<double, 3, 3> tensor;
} Interaction;

typedef struct {
  std::string type_i;
  std::string type_j;
  Vec3        r_ij;
  Mat3        J_ij;
} interaction_t;


typedef struct {
  int k;
  int a;
  int b;
  int c;
} inode_t;


inline bool operator <(const inode_t& x, const inode_t& y) {
    return std::tie(x.k, x.a, x.b, x.c) < std::tie(y.k, y.a, y.b, y.c);
}


typedef struct {
  inode_t node;
  Mat3    value;
} inode_pair_t;

void safety_check_distance_tolerance(const double &tolerance);
void generate_neighbour_list_from_file(std::ifstream &file, double energy_cutoff, bool use_symops, bool print_unfolded, InteractionList<Mat3>& neighbour_list);
void write_interaction_data(std::ostream &output, const std::vector<interaction_t> &data);
void write_neighbour_list(std::ostream &output, const InteractionList<Mat3> &list);

template <class T>
class InteractionList {
  public:
    typedef unsigned int            size_type;
    typedef std::map<size_type, T>          value_type;
    typedef value_type              reference;
    typedef const value_type&           const_reference;
    typedef value_type*                 pointer;
    typedef const value_type*         const_pointer;
    typedef pointer                             iterator;
    typedef const_pointer                       const_iterator;

    InteractionList()
      : list() {};

    InteractionList(size_type n)
      : list(n) {};

    ~InteractionList() {};

    std::pair<typename value_type::iterator,bool>
  insert(size_type i, size_type j, const T &value);
    
    void resize(size_type size);

    size_type size() const;

    const_reference interactions(size_type i) const;

          reference operator[] (const size_type i);
    const_reference operator[] (const size_type i) const;

  private:

    std::vector<value_type> list;
};

//---------------------------------------------------------------------

template <class T>
std::pair<typename InteractionList<T>::value_type::iterator,bool>
InteractionList<T>::insert(size_type i, size_type j, const T &value) {
  if (i >= list.size()) {
    list.resize(i+1);
  }

  return list[i].insert({j, value});
}

//---------------------------------------------------------------------

template <class T>
void 
InteractionList<T>::resize(size_type size) {
  list.resize(size);
}

//---------------------------------------------------------------------

template <class T>
typename InteractionList<T>::size_type
InteractionList<T>::size() const{
  return list.size();
}

//---------------------------------------------------------------------

template <class T>
typename InteractionList<T>::const_reference
InteractionList<T>::interactions(size_type i) const {
  return list[i];
}

//---------------------------------------------------------------------

template <class T>
typename InteractionList<T>::const_reference
InteractionList<T>::operator[](const size_type i) const {
  return list[i];
}

//---------------------------------------------------------------------

template <class T>
typename InteractionList<T>::reference
InteractionList<T>::operator[](const size_type i) {
  return list[i];
}

#endif // JAMS_CORE_INTERACTIONS_H