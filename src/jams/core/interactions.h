#ifndef JAMS_CORE_INTERACTIONS_H
#define JAMS_CORE_INTERACTIONS_H

#include <iosfwd>
#include <vector>
#include <tuple>
#include <utility>
#include <map>
#include <tuple>
#include <cassert>

#include <libconfig.h++>

#include "jams/core/types.h"
#include "jams/containers/mat3.h"
#include "jams/helpers/utils.h"
#include "jams/containers/unordered_vector_set.h"
#include "jams/containers/vector_set.h"
#include "jams/containers/interaction_list.h"

template <class T>
class InteractionList;


inline InteractionFileFormat interaction_file_format_from_string(const std::string s) {
  if (capitalize(s) == "JAMS") return InteractionFileFormat::JAMS;
  if (capitalize(s) == "KKR") return InteractionFileFormat::KKR;
  throw std::runtime_error("Unknown exchange file format");
}

// Exchange can be specified isotropic (1 scalar) or a full tensor (9 scalars)
enum class InteractionType {UNDEFINED, SCALAR, TENSOR};

enum class InteractionChecks {
    kNoZeroMotifNeighbourCount,    /// < Check no motif positions have zero neighbours
    kIdenticalMotifNeighbourCount, /// < Check if all motif positions have the same number of neighbours
    kIdenticalMotifTotalExchange   /// < Check all motif positions have the same total exchange value (based on diagonal part of exchange)
};

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
    int         basis_site_i = -1;
    int         basis_site_j = -1;
    Vec3        r_ij = {{0.0, 0.0, 0.0}}; // interaction vector (cartesian)
    Mat3        J_ij = kZeroMat3;
    std::string type_i = "NOTYPE";
    std::string type_j = "NOTYPE";
};

struct IntegerInteractionData {
    int         basis_site_i = -1;
    int         basis_site_j = -1;
    Vec3i       T_ij = {{0, 0, 0}};   // offset in unitcells
    Mat3        J_ij = kZeroMat3;
    std::string type_i = "NOTYPE";
    std::string type_j = "NOTYPE";
};

InteractionFileDescription
discover_interaction_file_format(std::ifstream &file);

std::vector<InteractionData>
interactions_from_file(std::ifstream &file, const InteractionFileDescription& desc);

jams::InteractionList<Mat3, 2>
neighbour_list_from_interactions(std::vector<InteractionData> &interactions);

jams::InteractionList<Mat3, 2>
generate_neighbour_list(std::ifstream &file,
        CoordinateFormat coord_format = CoordinateFormat::CARTESIAN, bool use_symops = true,
        double energy_cutoff = 0.0, double radius_cutoff = 0.0, std::vector<InteractionChecks> checks = {
    InteractionChecks::kNoZeroMotifNeighbourCount, InteractionChecks::kIdenticalMotifNeighbourCount, InteractionChecks::kIdenticalMotifTotalExchange});

jams::InteractionList<Mat3, 2>
generate_neighbour_list(libconfig::Setting& settings,
        CoordinateFormat coord_format = CoordinateFormat::CARTESIAN, bool use_symops = true,
        double energy_cutoff = 0.0, double radius_cutoff = 0.0, std::vector<InteractionChecks> checks = {
    InteractionChecks::kNoZeroMotifNeighbourCount, InteractionChecks::kIdenticalMotifNeighbourCount, InteractionChecks::kIdenticalMotifTotalExchange});

void
safety_check_distance_tolerance(const double &tolerance);

/// Check that for every interaction in the list there is a corresponding interaction which has the correct symmetries
/// J_ij = (J_ji)^T and r_ij = -r_ji
void
check_interaction_list_symmetry(const std::vector<InteractionData> &interactions);

void
write_interaction_data(std::ostream &output, const std::vector<InteractionData> &data,
                       CoordinateFormat coord_format);

void
write_neighbour_list(std::ostream &output, const jams::InteractionList<Mat3, 2> &list);

template <class T>
class InteractionList {
  public:
    typedef unsigned int            size_type;
    typedef std::map<size_type, size_type> value_type;
    typedef value_type              reference;
    typedef const value_type&           const_reference;
    typedef value_type*                 pointer;
    typedef const value_type*         const_pointer;
    typedef pointer                             iterator;
    typedef const_pointer                       const_iterator;
    typedef typename std::map<size_type, size_type>::value_type pair_type;

    InteractionList()
      : interactions_() {};

    InteractionList(size_type n)
      : interactions_(n) {};

    std::pair<typename value_type::iterator,bool>
    insert(size_type i, size_type j, const T &value);

    bool exists(size_type i, size_type j);
    
    void resize(size_type size);

    size_type size() const;

    size_type num_interactions() const;
    size_type num_interactions(const int i) const;

    const T& table(size_type i) const {
      return values_[i];
    }

    const_reference interactions(size_type i) const;

          reference operator[] (const size_type i);
    const_reference operator[] (const size_type i) const;

  private:
    jams::UnorderedVectorSet<T> values_;
    std::vector<value_type> interactions_;
};

template <class T>
std::pair<typename InteractionList<T>::value_type::iterator,bool>
InteractionList<T>::insert(size_type i, size_type j, const T &value) {
  if (i >= interactions_.size()) {
    interactions_.resize(i + 1);
  }

  auto it = values_.insert(value);
  auto k = std::distance(values_.begin(), it.first);
  return interactions_[i].insert({j, k});
}

template <class T>
void 
InteractionList<T>::resize(size_type size) {
  interactions_.resize(size);
}

template <class T>
typename InteractionList<T>::size_type
InteractionList<T>::size() const{
  return interactions_.size();
}

template <class T>
typename InteractionList<T>::size_type
InteractionList<T>::num_interactions(const int i) const{
  return interactions_[i].size();
}

template <class T>
typename InteractionList<T>::size_type
InteractionList<T>::num_interactions() const{
  size_type total = 0;

  for (auto map : interactions_) {
    total += map.size();
  }
  return total;
}

template <class T>
typename InteractionList<T>::const_reference
InteractionList<T>::interactions(size_type i) const {
  return interactions_[i];
}

template <class T>
typename InteractionList<T>::const_reference
InteractionList<T>::operator[](const size_type i) const {
  return interactions_[i];
}

template <class T>
typename InteractionList<T>::reference
InteractionList<T>::operator[](const size_type i) {
  return interactions_[i];
}

template<class T>
bool InteractionList<T>::exists(InteractionList::size_type i, InteractionList::size_type j) {
  return interactions_[i].count(j);
}

#endif // JAMS_CORE_INTERACTIONS_H
