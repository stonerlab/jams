//
// Created by Joe Barker on 2017/11/20.
//

#ifndef JAMS_UNITCELL_H
#define JAMS_UNITCELL_H

#include <array>
#include <vector>
#include <cassert>

#include "jams/core/types.h"


template <class Type>
class UnitCell {
public:
    UnitCell() = default;

    explicit UnitCell(jams::Mat<double, 3, 3> vectors,
                    std::vector<jams::Vec<double, 3>> positions = std::vector<jams::Vec<double, 3>>(),
                    std::vector<Type> types = std::vector<Type>());

    UnitCell(const jams::Vec<double, 3> &a, const jams::Vec<double, 3> &b, const jams::Vec<double, 3> &c, std::vector<jams::Vec<double, 3>> &positions,
             std::vector<Type> &types);

    void insert(const Type &type, const jams::Vec<double, 3> &position);

    jams::Vec<double, 3> a() const;
    jams::Vec<double, 3> b() const;
    jams::Vec<double, 3> c() const;

    size_t num_atoms() const;

    const jams::Mat<double, 3, 3>& vectors() const;
    const jams::Mat<double, 3, 3>& inverse_vectors() const;
    const std::vector<jams::Vec<double, 3>>& positions() const;
    const std::vector<Type>& types() const;

protected:
    jams::Mat<double, 3, 3>  matrix_ = {1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0};
    jams::Mat<double, 3, 3>  inverse_matrix_ = {1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0};
    std::vector<jams::Vec<double, 3>> positions_;
    std::vector<Type> types_;
};

template <class Type>
double volume(const UnitCell<Type>& unitcell);


template<class Type>
UnitCell<Type>::UnitCell(jams::Mat<double, 3, 3> vectors, std::vector<jams::Vec<double, 3>> positions, std::vector<Type> types)
: matrix_(vectors),
  inverse_matrix_(inverse(matrix_)),
  positions_(std::move(positions)),
  types_(std::move(types))
{
  assert(positions_.size() == types_.size());
}

template<class Type>
UnitCell<Type>::UnitCell(const jams::Vec<double, 3> &a, const jams::Vec<double, 3> &b, const jams::Vec<double, 3> &c, std::vector<jams::Vec<double, 3>> &positions,
                         std::vector<Type> &types) : UnitCell(matrix_from_cols(a, b, c), positions,types) {};

template<class Type>
void UnitCell<Type>::insert(const Type &type, const jams::Vec<double, 3> &position) {
  positions_.emplace_back(position);
  types_.emplace_back(type);
}

template<class Type>
jams::Vec<double, 3> UnitCell<Type>::a() const {
  return {matrix_[0][0], matrix_[1][0], matrix_[2][0]};
}

template<class Type>
jams::Vec<double, 3> UnitCell<Type>::b() const {
  return {matrix_[0][1], matrix_[1][1], matrix_[2][1]};
}

template<class Type>
jams::Vec<double, 3> UnitCell<Type>::c() const {
  return {matrix_[0][2], matrix_[1][2], matrix_[2][2]};
}

template<class Type>
size_t UnitCell<Type>::num_atoms() const {
  return positions_.size();
}

template<class Type>
const std::vector<jams::Vec<double, 3>>& UnitCell<Type>::positions() const {
  return positions_;
}

template<class Type>
const std::vector<Type>& UnitCell<Type>::types() const {
  return types_;
}

template<class Type>
const jams::Mat<double, 3, 3>& UnitCell<Type>::vectors() const {
  return matrix_;
}

template<class Type>
const jams::Mat<double, 3, 3> &UnitCell<Type>::inverse_vectors() const {
  return inverse_matrix_;
}

template <class Type>
inline double volume(const UnitCell<Type> &u) {
  return jams::scalar_triple_product(u.a(), u.b(), u.c());
}

#endif //JAMS_UNITCELL_H
