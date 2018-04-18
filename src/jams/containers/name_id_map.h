//
// Created by Joe Barker on 2017/12/06.
//

#ifndef JAMS_NAME_ID_MAP_H
#define JAMS_NAME_ID_MAP_H

#include <vector>
#include "jams/containers/bimap.h"

template <class T>
class NameIdMap {
public:
    NameIdMap() = default;

    void insert(const std::string &name, const T& x);

    const T& operator[](const std::size_t& id) const;
    const T& operator[](const std::string& name) const;

    const std::string& name(const std::size_t& id) const;
    const std::size_t& id(const std::string& name) const;

    bool contains(const std::size_t& id) const;
    bool contains(const std::string& name) const;

    std::size_t size() const;
    void clear();

private:
    std::size_t uid = 0;
    std::vector<T>                  data_;
    Bimap<std::size_t, std::string> bimap_;
};


template<class T>
void NameIdMap<T>::insert(const std::string &name, const T &x) {
  data_.emplace_back(x);
  bimap_.insert(uid, name);
  uid++;
}

template<class T>
const T &NameIdMap<T>::operator[](const std::size_t &id) const {
  return data_[id];
}

template<class T>
const T &NameIdMap<T>::operator[](const std::string &name) const {
  return data_[this->id(name)];
}

template<class T>
const std::string &NameIdMap<T>::name(const std::size_t &id) const {
  return bimap_.left(id);
}

template<class T>
const std::size_t &NameIdMap<T>::id(const std::string &name) const {
  return bimap_.right(name);
}

template<class T>
std::size_t NameIdMap<T>::size() const {
  return data_.size();
}

template<class T>
void NameIdMap<T>::clear() {
  data_.clear();
  bimap_.clear();
  uid = 0;
}

template<class T>
bool NameIdMap<T>::contains(const std::size_t &id) const {
  return bimap_.left_contains(id);
}

template<class T>
bool NameIdMap<T>::contains(const std::string &name) const {
  return bimap_.right_contains(name);
}


#endif //JAMS_NAME_ID_MAP_H
