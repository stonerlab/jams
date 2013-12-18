#include <new>
#include <iostream>
#include <vector>
#include "./Array.h"
#include "../math/Vec.h"
#include "../sys/sys_defines.h"


namespace jbLib{
  template < typename type, typename index >
    class Sparsematrix < type, 3, index>
    {
      public:

        // default constructor
        Sparsematrix(const index sx=0, const index sy=0, const index sz=0)
          : finalized(false),
          nNonZero(0),
          dimx(sx),
          dimy(sy),
          dimz(sz),
          map(),
          hashTable(),
          valueTable(),
          indexTable(),
          zeroValue(0)
      { }

        // copy constructor
        Sparsematrix(const Sparsematrix< type, 3, index >& other)
          : finalized(other.finalized),
          nNonZero(other.nNonZero),
          dimx(other.dimx),
          dimy(other.dimy),
          dimz(other.dimz),
          map(other.map),
          hashTable(other.hashTable),
          valueTable(other.valueTable),
          indexTable(other.indexTable),
          zeroValue(0)
      { }

        ~Sparsematrix(){
        }

        JB_INLINE void insert(const index i, const index j, const index k, const type val);
        JB_INLINE void getIndex(const index i, index& x, index& y, index& z) const;
        JB_INLINE Vec<int> getVec(const index i) const;
        JB_INLINE void finalize();

        //Normal Access Operations
        const type& JB_RESTRICT operator()(const index &i, const index &j, const index &k) const{ 
          assert( checkFinalized() );
          assert( checkBounds(i,j,k) );

          typename std::vector<index>::const_iterator it;

          it = binary_find( hashTable.begin(),hashTable.end(), hashFunc(i,j,k) );
          if( it != hashTable.end() ){
            return valueTable[ it - hashTable.begin() ];
          }
          return zeroValue;
        }

        JB_FORCE_INLINE  const type& operator[](const index i) const {
          assert( checkFinalized() );
          assert( i < nNonZero );
          return valueTable[i];
        }

        JB_FORCE_INLINE  type& operator[](const index i) {
          assert( checkFinalized() );
          assert( i < nNonZero );
          return valueTable[i];
        }

        JB_FORCE_INLINE index nonZeros() const {
          return nNonZero;
        }

        JB_FORCE_INLINE  type* JB_RESTRICT ptr() {
          assert( checkFinalized() );
          return &valueTable[0];
        }

        JB_FORCE_INLINE  const type* JB_RESTRICT ptr() const {
          assert( checkFinalized() );
          return &valueTable[0];
        }

        JB_FORCE_INLINE const index& JB_RESTRICT sizex() const { return dimx; }
        JB_FORCE_INLINE const index& JB_RESTRICT sizey() const { return dimy; }
        JB_FORCE_INLINE const index& JB_RESTRICT sizez() const { return dimz; }

        friend void swap(Sparsematrix< type, 3, index >& first, Sparsematrix< type, 3, index >& second) // nothrow
        { 
          std::swap(first.finalized, second.finalized);
          std::swap(first.nNonZero,second.nNonZero);
          std::swap(first.dimx,second.dimx);
          std::swap(first.dimy,second.dimy);
          std::swap(first.dimz,second.dimz);
          std::swap(first.map,second.map);
          std::swap(first.hashTable,second.hashTable);
          std::swap(first.valueTable,second.valueTable);
          std::swap(first.indexTable,second.indexTable);
        }

        Sparsematrix< type, 3, index >& operator=(Sparsematrix< type, 3, index > rhs){
          swap(*this, rhs);
          return *this;
        }

      private:

        JB_INLINE index hashFunc(const index i, const index j, const index k) const{
          return (i*dimy + j)*dimz + k;
        }

        JB_INLINE void invHashFunc(const index hash, index &i, index &j, index &k) const{
          i = hash/(dimy*dimz);
          j = (hash - (dimy*dimz)*i)/(dimz);
          k = (hash - dimz*(j + dimy*i));
          assert( hashFunc(i,j,k) == hash );
        }

        JB_INLINE bool checkFinalized() const{ return finalized;}

        JB_INLINE bool checkBounds(const index& i, const index& j, const index& k) const{
          if( i < dimx && j < dimy && k < dimz){
            return true;
          } else {
            return false;
          }
        }



        bool  finalized;
        index nNonZero;
        index dimx;
        index dimy;
        index dimz;
        std::vector< std::pair<index,type> > map;
        std::vector< index > hashTable;
        std::vector< type  > valueTable;
        Array<index,2> indexTable;
        const type  zeroValue;
    };

  template < typename type, typename index >
    JB_INLINE void Sparsematrix<type,3,index>::insert(const index i, const index j, const index k, const type val){

      assert(i < dimx); assert( !(i < 0) );
      assert(j < dimy); assert( !(j < 0) );
      assert(k < dimz); assert( !(k < 0) );

      map.push_back( std::pair<index,type>( hashFunc(i,j,k), val ) );
      nNonZero++;

    }

  template < typename type, typename index >
    JB_INLINE void Sparsematrix<type,3,index>::getIndex(const index i, index& x, index& y, index& z) const{
      assert(checkFinalized());
      x = indexTable(i,0);
      y = indexTable(i,1);
      z = indexTable(i,2);
    }

  template < typename type, typename index >
    JB_INLINE Vec<int> Sparsematrix<type,3,index>::getVec(const index i) const{
      assert(checkFinalized());
      return Vec<int> (indexTable(i,0), indexTable(i,1), indexTable(i,2));
    }

  template < typename type, typename index >
    void Sparsematrix<type,3,index>::finalize(){

      JB_REGISTER index i;
      JB_REGISTER index lastHash;


      // sort by hash
      std::sort(map.begin(),map.end(),kv_pair_less<index,type>);

      // inset first element
      hashTable.push_back(map[0].first);
      lastHash = map[0].first;
      valueTable.push_back(map[0].second);

      for( i=1; i<map.size(); ++i){
        // if hashes are the same then combine the values
        if( map[i].first == lastHash ){
          map.back().second = map.back().second + map[i].second;
          nNonZero--;
        }else{
          hashTable.push_back(map[i].first);
          valueTable.push_back(map[i].second);

        }
        lastHash = map[i].first;
      }


      index x,y,z;

      indexTable.resize(nNonZero,3);

      for( i=0; i< hashTable.size(); ++i){
        invHashFunc(hashTable[i],x,y,z);
        indexTable(i,0) = x;
        indexTable(i,1) = y;
        indexTable(i,2) = z;
      }

      finalized = true;
    }


}
