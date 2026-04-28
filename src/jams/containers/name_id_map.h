//
// Created by Joe Barker on 2017/12/06.
//

#ifndef JAMS_NAME_ID_MAP_H
#define JAMS_NAME_ID_MAP_H

#include <stdexcept>
#include <string>
#include <unordered_map>
#include <utility>
#include <vector>

template <class T>
class NameIdMap {
public:
    using size_type = std::size_t;

    NameIdMap() = default;

    void insert(const std::string &name, const T& x);
    void insert(const std::string &name, T&& x);

    const T& operator[](const size_type& id) const;
    const T& operator[](const std::string& name) const;

    const std::string& name(const size_type& id) const;
    const size_type& id(const std::string& name) const;

    bool contains(const size_type& id) const;
    bool contains(const std::string& name) const;

    size_type size() const;
    void clear();

private:
    template <typename U>
    void insert_impl(const std::string& name, U&& x);

    std::vector<T> data_;
    std::vector<std::string> names_;
    std::unordered_map<std::string, size_type> ids_by_name_;
};


template<class T>
void NameIdMap<T>::insert(const std::string &name, const T &x) {
  insert_impl(name, x);
}

template<class T>
void NameIdMap<T>::insert(const std::string &name, T &&x) {
  insert_impl(name, std::move(x));
}

template<class T>
template <typename U>
void NameIdMap<T>::insert_impl(const std::string &name, U &&x) {
  if (contains(name)) {
    throw std::runtime_error("duplicate name inserted into NameIdMap: " + name);
  }

  const auto new_id = data_.size();
  data_.emplace_back(std::forward<U>(x));
  try {
    names_.push_back(name);
    try {
      ids_by_name_.emplace(names_.back(), new_id);
    } catch (...) {
      names_.pop_back();
      throw;
    }
  } catch (...) {
    data_.pop_back();
    throw;
  }
}

template<class T>
const T &NameIdMap<T>::operator[](const size_type &id) const {
  return data_.at(id);
}

template<class T>
const T &NameIdMap<T>::operator[](const std::string &name) const {
  return data_[this->id(name)];
}

template<class T>
const std::string &NameIdMap<T>::name(const size_type &id) const {
  return names_.at(id);
}

template<class T>
const typename NameIdMap<T>::size_type &NameIdMap<T>::id(const std::string &name) const {
  return ids_by_name_.at(name);
}

template<class T>
typename NameIdMap<T>::size_type NameIdMap<T>::size() const {
  return data_.size();
}

template<class T>
void NameIdMap<T>::clear() {
  data_.clear();
  names_.clear();
  ids_by_name_.clear();
}

template<class T>
bool NameIdMap<T>::contains(const size_type &id) const {
  return id < data_.size();
}

template<class T>
bool NameIdMap<T>::contains(const std::string &name) const {
  return ids_by_name_.find(name) != ids_by_name_.end();
}


#endif //JAMS_NAME_ID_MAP_H
