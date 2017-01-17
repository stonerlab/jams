// Copyright 2017 Joseph Barker. All rights reserved.

#ifndef JAMS_CORE_INTERATION_LIST_H
#define JAMS_CORE_INTERATION_LIST_H

#include <vector>
#include <map>

template <class T>
class InteractionList {
  public:
  	typedef unsigned int						size_type;
  	typedef std::map<size_type, T>     			value_type;
  	typedef value_type							reference;
  	typedef const value_type&   				const_reference;
  	typedef value_type*         				pointer;
  	typedef const value_type*					const_pointer;

  	InteractionList()
  		: list() {};

  	InteractionList(size_type n)
  		: list(n) {};

  	~InteractionList() {};

  	void insert(size_type i, size_type j, const T &value);
  	void resize(size_type size);

  	const_reference interactions(size_type i) const;

  private:

  	std::vector<value_type> list;
};

//---------------------------------------------------------------------

template <class T>
void
InteractionList<T>::insert(size_type i, size_type j, const T &value) {
	if (i >= list.size()) {
		list.resize(i+1);
	}

	list[i].insert({j, value});
}

//---------------------------------------------------------------------

template <class T>
void 
InteractionList<T>::resize(size_type size) {
	list.resize(size);
}

template <class T>
typename InteractionList<T>::const_reference
InteractionList<T>::interactions(size_type i) const {
	return list[i];
}


#endif // JAMS_CORE_INTERATION_LIST_H
