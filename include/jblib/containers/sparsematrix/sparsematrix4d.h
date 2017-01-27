#ifndef JBLIB_CONTAINERS_SPARSEMATRIX4D_H
#define JBLIB_CONTAINERS_SPARSEMATRIX4D_H

#include <new>
#include <iostream>
#include <vector>
#include "jblib/containers/sparsematrix/sparsematrix_template.h"
#include "jblib/containers/array.h"
#include "jblib/containers/vec.h"
#include "jblib/sys/define.h"

namespace jblib {

    template< typename valueType_ >
       inline bool kvCompareX(const std::pair<Vec4<int>,valueType_>&a, const std::pair<Vec4<int>,valueType_>&b){
            return ( a.first.x < b.first.x );
        }
    template< typename valueType_ >
       inline bool kvCompareY(const std::pair<Vec4<int>,valueType_>&a, const std::pair<Vec4<int>,valueType_>&b){
            return ( a.first.y < b.first.y );
        }
    template< typename valueType_ >
       inline bool kvCompareZ(const std::pair<Vec4<int>,valueType_>&a, const std::pair<Vec4<int>,valueType_>&b){
            return ( a.first.z < b.first.z );
        }
    template< typename valueType_ >
       inline bool kvCompareW(const std::pair<Vec4<int>,valueType_>&a, const std::pair<Vec4<int>,valueType_>&b){
            return ( a.first.w < b.first.w );
        }

    template <typename valueType_, typename indexType_>
    class Sparsematrix < valueType_, 4, indexType_>
    {
      public:

        // default constructor
        Sparsematrix(const indexType_ nx=0, const indexType_ ny=0, const indexType_ nz=0, const indexType_ nw=0)
          : finalized(false),
          nNonZero(0),
          dimx(nx),
          dimy(ny),
          dimz(nz),
          dimw(nw),
            map(),
            valueTable(),
            indexTable(),
            csrPointers(),
            zeroValue(0)
      { }

        // copy constructor
        Sparsematrix(const Sparsematrix< valueType_, 4, indexType_ >& other)
          : finalized(other.finalized),
          nNonZero(other.nNonZero),
          dimx(other.dimx),
          dimy(other.dimy),
          dimz(other.dimz),
          dimw(other.dimw),
          map(other.map),
          valueTable(other.valueTable),
          indexTable(other.indexTable),
          csrPointers(other.csrPointers),
          zeroValue(0)
      { }

        ~Sparsematrix(){
        }

        void resize(const indexType_ nx, const indexType_ ny, const indexType_ nz, const indexType_ nw);

        void insert(const indexType_ i, const indexType_ j, const indexType_ k, const indexType_ l, const valueType_ val);
        void getIndex(const indexType_ i, indexType_& x, indexType_& y, indexType_& z, indexType_& w) const;
        void finalize();

        //Normal Access Operations
        const valueType_& restrict operator()(const indexType_ &i, const indexType_ &j, const indexType_ &k, const indexType_ &w) const;

        const valueType_& operator[](const indexType_ i) const;
        valueType_& operator[](const indexType_ i);

        indexType_ nonZeros() const;

        const valueType_* restrict valueData() const;
              valueType_* restrict valueData();
        const indexType_* restrict indexData() const;
              indexType_* restrict indexData();
        const indexType_* restrict csrData() const;
              indexType_* restrict csrData();

        const indexType_& restrict sizex() const;
        const indexType_& restrict sizey() const;
        const indexType_& restrict sizez() const;
        const indexType_& restrict sizew() const;

        double calculateMemoryUsage() const;

        template <typename fvalueType_, typename findexType_>
        friend void swap(Sparsematrix< fvalueType_, 4, findexType_ >& first, Sparsematrix< fvalueType_, 4, findexType_ >& second); // nothrow

        Sparsematrix<valueType_,4,indexType_>& operator=(Sparsematrix<valueType_,4,indexType_> rhs);

      private:

        bool isFinalized() const;
        bool isWithinBounds(const indexType_& i, const indexType_& j, const indexType_& k, const indexType_ &l) const;

        void createCsrArray();

        bool  finalized;
        indexType_ nNonZero;
        indexType_ dimx;
        indexType_ dimy;
        indexType_ dimz;
        indexType_ dimw;
        std::vector< std::pair<Vec4<int>,valueType_> > map;
        std::vector< valueType_  > valueTable;
        Array<indexType_,2> indexTable;
        std::vector< indexType_ > csrPointers;
        const valueType_  zeroValue;
    };

  template < typename valueType_, typename indexType_>
    void Sparsematrix<valueType_,4,indexType_>::resize(const indexType_ nx, const indexType_ ny, const indexType_ nz, const indexType_ nw){

      finalized = false;

      nNonZero = 0;

      dimx = nx;
      dimy = ny;
      dimz = nz;
      dimw = nw;

      map.clear();
      valueTable.clear();
      indexTable.resize(0,0);

    }

  template < typename valueType_, typename indexType_>
    inline void Sparsematrix<valueType_,4,indexType_>::insert(const indexType_ i, const indexType_ j, const indexType_ k, const indexType_ l, const valueType_ val){

      assert(i < dimx); assert( !(i < 0) );
      assert(j < dimy); assert( !(j < 0) );
      assert(k < dimz); assert( !(k < 0) );
      assert(l < dimw); assert( !(l < 0) );

      map.push_back( std::pair<Vec4<int>,valueType_>( Vec4<int>(i,j,k,l), val ) );
      nNonZero++;

    }

  template < typename valueType_, typename indexType_>
    inline void Sparsematrix<valueType_,4,indexType_>::getIndex(const indexType_ i, indexType_& x, indexType_& y, indexType_& z, indexType_& w) const{
      assert(isFinalized());
      x = indexTable(i,0);
      y = indexTable(i,1);
      z = indexTable(i,2);
      w = indexTable(i,3);
    }

  template < typename valueType_, typename indexType_>
    void Sparsematrix<valueType_,4,indexType_>::finalize(){

      indexType_ i;
      Vec4<int> lastVec;


      // sort by hash
      std::stable_sort(map.begin(),map.end(),kvCompareW<valueType_>);
      std::stable_sort(map.begin(),map.end(),kvCompareZ<valueType_>);
      std::stable_sort(map.begin(),map.end(),kvCompareY<valueType_>);
      std::stable_sort(map.begin(),map.end(),kvCompareX<valueType_>);

      indexTable.resize(nNonZero,4);

      // inset first element
      indexTable(0,0) = map[0].first.x;
      indexTable(0,1) = map[0].first.y;
      indexTable(0,2) = map[0].first.z;
      indexTable(0,3) = map[0].first.w;
      lastVec = map[0].first;
      valueTable.push_back(map[0].second);


      for( i=1; i<map.size(); ++i){
        // if hashes are the same then combine the values
        if( map[i].first == lastVec ){
          map.back().second = map.back().second + map[i].second;
          nNonZero--;
        }else{
          indexTable(i,0) = map[i].first.x;
          indexTable(i,1) = map[i].first.y;
          indexTable(i,2) = map[i].first.z;
          indexTable(i,3) = map[i].first.w;
          valueTable.push_back(map[i].second);
        }
        lastVec = map[i].first;
      }

      // NOTE: should we resize indexTable here?

      createCsrArray();
      finalized = true;
    }


  //template < typename valueType_, typename indexType_>
    //const valueType_& restrict Sparsematrix<valueType_,4,indexType_>::operator()(const indexType_ &i, const indexType_ &j, const indexType_ &k, const indexType_ &l) const{
      //assert( isFinalized() );
      //assert( isWithinBounds(i,j,k,l) );

      //typename std::vector<hashType_>::const_iterator it;

      //it = binary_find( hashTable.begin(),hashTable.end(), hashFunc(i,j,k,l) );
      //if( it != hashTable.end() ){
        //return valueTable[ it - hashTable.begin() ];
      //}
      //return zeroValue;
    //}

  template < typename valueType_, typename indexType_>
    inline  const valueType_& Sparsematrix<valueType_,4,indexType_>::operator[](const indexType_ i) const {
      assert( isFinalized() );
      assert( i < nNonZero );
      return valueTable[i];
    }

  template < typename valueType_, typename indexType_>
    inline  valueType_& Sparsematrix<valueType_,4,indexType_>::operator[](const indexType_ i) {
      assert( isFinalized() );
      assert( i < nNonZero );
      return valueTable[i];
    }

  template < typename valueType_, typename indexType_>
    inline indexType_ Sparsematrix<valueType_,4,indexType_>::nonZeros() const {
      return nNonZero;
    }

  template < typename valueType_, typename indexType_>
    inline  valueType_* restrict Sparsematrix<valueType_,4,indexType_>::valueData() {
      assert( isFinalized() );
      return &valueTable[0];
    }

  template < typename valueType_, typename indexType_>
    inline  const valueType_* restrict Sparsematrix<valueType_,4,indexType_>::valueData() const {
      assert( isFinalized() );
      return &valueTable[0];
    }

  template < typename valueType_, typename indexType_>
    inline  indexType_* restrict Sparsematrix<valueType_,4,indexType_>::indexData() {
      assert( isFinalized() );
      return indexTable.data();
    }

  template < typename valueType_, typename indexType_>
    inline  const indexType_* restrict Sparsematrix<valueType_,4,indexType_>::indexData() const {
      assert( isFinalized() );
      return indexTable.data();
    }


  template < typename valueType_, typename indexType_>
    inline  indexType_* restrict Sparsematrix<valueType_,4,indexType_>::csrData() {
      assert( isFinalized() );
      return &csrPointers[0];
    }

  template < typename valueType_, typename indexType_>
    inline  const indexType_* restrict Sparsematrix<valueType_,4,indexType_>::csrData() const {
      assert( isFinalized() );
      return &csrPointers[0];
    }
  template < typename valueType_, typename indexType_>
    inline const indexType_& restrict Sparsematrix<valueType_,4,indexType_>::sizex() const { return dimx; }
  template < typename valueType_, typename indexType_>
    inline const indexType_& restrict Sparsematrix<valueType_,4,indexType_>::sizey() const { return dimy; }
  template < typename valueType_, typename indexType_>
    inline const indexType_& restrict Sparsematrix<valueType_,4,indexType_>::sizez() const { return dimz; }
  template < typename valueType_, typename indexType_>
    inline const indexType_& restrict Sparsematrix<valueType_,4,indexType_>::sizew() const { return dimw; }

  template < typename valueType_, typename indexType_>
    inline void swap(Sparsematrix<valueType_,4,indexType_>& first, Sparsematrix<valueType_,4,indexType_>& second) // nothrow
    {
      std::swap(first.finalized, second.finalized);
      std::swap(first.nNonZero,second.nNonZero);
      std::swap(first.dimx,second.dimx);
      std::swap(first.dimy,second.dimy);
      std::swap(first.dimz,second.dimz);
      std::swap(first.dimw,second.dimw);
      std::swap(first.map,second.map);
      std::swap(first.valueTable,second.valueTable);
      std::swap(first.indexTable,second.indexTable);
    }

  template < typename valueType_, typename indexType_>
    inline Sparsematrix<valueType_,4,indexType_>& Sparsematrix<valueType_,4,indexType_>::operator=(Sparsematrix<valueType_,4,indexType_> rhs){
      swap(*this, rhs);
      return *this;
    }


  template < typename valueType_, typename indexType_>
    inline bool Sparsematrix<valueType_,4,indexType_>::isFinalized() const{ return finalized;}

  template < typename valueType_, typename indexType_>
    inline bool Sparsematrix<valueType_,4,indexType_>::isWithinBounds(const indexType_& i, const indexType_& j, const indexType_& k, const indexType_& l) const{
      if( i < dimx && j < dimy && k < dimz && l < dimw){
        return true;
      } else {
        return false;
      }
    }

  template < typename valueType_, typename indexType_>
    inline double Sparsematrix<valueType_,4,indexType_>::calculateMemoryUsage() const{
      return ( indexTable.size(0)*indexTable.size(1)*sizeof(indexType_) + map.size()*sizeof(Vec4<int>) + map.size()*sizeof(valueType_) + valueTable.size()*sizeof(valueType_) )/(1024.0*1024.0);
    }

  // This routine assumes that the tables are ordered by i,j,k,l so this
  // routine should only be called after finalize()
  template < typename valueType_, typename indexType_>
    inline void Sparsematrix<valueType_,4,indexType_>::createCsrArray() {

      indexType_ n,count,lastRow,currentRow;

      csrPointers.resize(dimx+1);

      lastRow = -1;
      count = 0;
      if(nNonZero > 0){
          for(n=0; n<nNonZero; ++n){
            currentRow = indexTable(n,0);

            if(currentRow != lastRow){
                assert(count < (dimx+1));
                csrPointers[count] = n;
                count++;
            }
            lastRow = currentRow;
          }
          csrPointers[dimx] = nNonZero;
      }
    }
}

#endif  // JBLIB_CONTAINERS_SPARSEMATRIX4D_H
