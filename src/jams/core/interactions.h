#ifndef JAMS_CORE_INTERACTIONS_H
#define JAMS_CORE_INTERACTIONS_H

#include <iosfwd>
#include <vector>
#include <tuple>
#include <utility>
#include <map>

#include <libconfig.h++>

#include "jams/core/types.h"
#include "jams/containers/mat3.h"
#include "jams/helpers/utils.h"



template <class T>
class InteractionList;

//
// JAMS format:
// typename_i typename_j rx ry rz Jxx Jxy Jxz Jyx Jyy Jyz Jzx Jzy Jzz
//
// KKR format:
// unitcell_pos_i unitcell_pos_j rx ry rz Jxx Jxy Jxz Jyx Jyy Jyz Jzx Jzy Jzz
//
enum class InteractionFileFormat {UNDEFINED, JAMS, KKR};

inline InteractionFileFormat interaction_file_format_from_string(const std::string s) {
  if (capitalize(s) == "JAMS") return InteractionFileFormat::JAMS;
  if (capitalize(s) == "KKR") return InteractionFileFormat::KKR;
  throw std::runtime_error("Unknown exchange file format");
}

// Exchange can be specified isotropic (1 scalar) or a full tensor (9 scalars)
enum class InteractionType {UNDEFINED, SCALAR, TENSOR};

struct InteractionFileDescription {
    InteractionFileDescription() = default;

    InteractionFileDescription(InteractionFileFormat t, InteractionType d)
      : type(t), dimension(d) {};

    bool operator==(InteractionFileDescription rhs) {
      return rhs.type == type && rhs.dimension == dimension;
    }

    InteractionFileFormat  type       = InteractionFileFormat::UNDEFINED;
    InteractionType dimension = InteractionType::UNDEFINED;
};

struct InteractionData {
    int         unit_cell_pos_i = -1;
    int         unit_cell_pos_j = -1;
    Vec3        r_ij = {{0.0, 0.0, 0.0}}; // interaction vector (cartesian)
    Mat3        J_ij = kZeroMat3;
    std::string type_i = "NOTYPE";
    std::string type_j = "NOTYPE";
};

struct IntegerInteractionData {
    int         unit_cell_pos_i = -1;
    int         unit_cell_pos_j = -1;
    Vec3i       u_ij = {{0, 0, 0}};   // offset in unitcells
    Mat3        J_ij = kZeroMat3;
    std::string type_i = "NOTYPE";
    std::string type_j = "NOTYPE";
};

InteractionFileDescription
discover_interaction_file_format(std::ifstream &file);

std::vector<InteractionData>
interactions_from_file(std::ifstream &file, const InteractionFileDescription& desc);

InteractionList<Mat3>
neighbour_list_from_interactions(std::vector<InteractionData> &interactions,
        CoordinateFormat coord_format, bool use_symops, double energy_cutoff, double radius_cutoff);

InteractionList<Mat3>
generate_neighbour_list(std::ifstream &file,
        CoordinateFormat coord_format = CoordinateFormat::CARTESIAN, bool use_symops = true,
        double energy_cutoff = 0.0, double radius_cutoff = 0.0);

InteractionList<Mat3>
generate_neighbour_list(libconfig::Setting& settings,
        CoordinateFormat coord_format = CoordinateFormat::CARTESIAN, bool use_symops = true,
        double energy_cutoff = 0.0, double radius_cutoff = 0.0);

void
safety_check_distance_tolerance(const double &tolerance);

void
write_interaction_data(std::ostream &output, const std::vector<InteractionData> &data,
                       CoordinateFormat coord_format);

void
write_neighbour_list(std::ostream &output, const InteractionList<Mat3> &list);


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
    typedef typename std::map<size_type, T>::value_type pair_type;

    InteractionList()
      : list() {};

    InteractionList(size_type n)
      : list(n) {};

    ~InteractionList() {};

    std::pair<typename value_type::iterator,bool>
    insert(size_type i, size_type j, const T &value);

    bool exists(size_type i, size_type j);
    
    void resize(size_type size);

    size_type size() const;

    size_type num_interactions() const;
    size_type num_interactions(const int i) const;

    const_reference interactions(size_type i) const;

          reference operator[] (const size_type i);
    const_reference operator[] (const size_type i) const;

  private:
    std::vector<value_type> list;
};

template <class T>
std::pair<typename InteractionList<T>::value_type::iterator,bool>
InteractionList<T>::insert(size_type i, size_type j, const T &value) {
  if (i >= list.size()) {
    list.resize(i+1);
  }

  return list[i].insert({j, value});
}

template <class T>
void 
InteractionList<T>::resize(size_type size) {
  list.resize(size);
}

template <class T>
typename InteractionList<T>::size_type
InteractionList<T>::size() const{
  return list.size();
}

template <class T>
typename InteractionList<T>::size_type
InteractionList<T>::num_interactions(const int i) const{
  return list[i].size();
}

template <class T>
typename InteractionList<T>::size_type
InteractionList<T>::num_interactions() const{
  size_type total = 0;

  for (auto map : list) {
    total += map.size();
  }
  return total;
}

template <class T>
typename InteractionList<T>::const_reference
InteractionList<T>::interactions(size_type i) const {
  return list[i];
}

template <class T>
typename InteractionList<T>::const_reference
InteractionList<T>::operator[](const size_type i) const {
  return list[i];
}

template <class T>
typename InteractionList<T>::reference
InteractionList<T>::operator[](const size_type i) {
  return list[i];
}

template<class T>
bool InteractionList<T>::exists(InteractionList::size_type i, InteractionList::size_type j) {
  return list[i].find(j) != list[i].end();
}

#endif // JAMS_CORE_INTERACTIONS_H
