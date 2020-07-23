//
// Created by Joe Barker on 2017/11/20.
//

#ifndef JAMS_BIMAP_H
#define JAMS_BIMAP_H

#include <map>

// based on https://stackoverflow.com/questions/21760343/is-there-a-more-efficient-implementation-for-a-bidirectional-map

template <typename A, typename B>
class Bimap {
public:
    Bimap() = default;

    void insert(const A &a, const B &b);

    const B& left(const A& a) const;
    const A& right(const B& b) const;

    bool left_contains(const A& a) const;
    bool right_contains(const B& b) const;

    std::size_t size() const;
    void clear();

private:
    std::map<std::size_t, A> mapA;
    std::map<std::size_t, B> mapB;
};

template<typename A, typename B>
void Bimap<A, B>::insert(const A &a, const B &b) {
  assert(mapA.size() == mapB.size());
  std::size_t hashA = std::hash<A>()(a);
  std::size_t hashB = std::hash<B>()(b);
  if (mapA.count(hashA) || mapB.count(hashB)) {
    // a or b already exist in the bimap
    return;
  }
  mapA.insert({hashB, a});
  mapB.insert({hashA, b});
}

template<typename A, typename B>
const B &Bimap<A, B>::left(const A &a) const {
  return mapB.at(std::hash<A>()(a));
}

template<typename A, typename B>
const A &Bimap<A, B>::right(const B &b) const {
  return mapA.at(std::hash<B>()(b));
}

template<typename A, typename B>
std::size_t Bimap<A, B>::size() const {
  assert(mapA.size() == mapB.size());
  return mapA.size();
}

template<typename A, typename B>
void Bimap<A, B>::clear() {
  mapA.clear();
  mapB.clear();
}

template<typename A, typename B>
bool Bimap<A, B>::left_contains(const A &a) const {
  const auto search = mapB.find(std::hash<A>()(a));
  return search != mapB.end();
}

template<typename A, typename B>
bool Bimap<A, B>::right_contains(const B &b) const {
  const auto search = mapA.find(std::hash<B>()(b));
  return search != mapA.end();}

#endif //JAMS_BIMAP_H
