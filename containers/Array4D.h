#include <new>
#include <iostream>

#ifdef BOUNDSCHECK
#include <stdexcept>
#include <sstream>
#endif

namespace jbLib {
  template <typename Tp_, typename Idx_>
    class Array <Tp_,4,Idx_> {
      public:

        Array(const Idx_ sx=0, const Idx_ sy=0, const Idx_ sz=0, const Idx_ sw=0);
        Array(const Idx_ sx, const Idx_ sy, const Idx_ sz, const Idx_ sw, const Tp_ ival);

        ~Array();
        Array(const Array<Tp_,4,Idx_>& other);
        Array<Tp_,4,Idx_>& operator=(Array<Tp_,4,Idx_> rhs);

        ALIGNTYPE64       Tp_& JB_RESTRICT operator()
          (const Idx_ i, const Idx_ j, const Idx_ k, const Idx_ l);
        ALIGNTYPE64 const Tp_& JB_RESTRICT operator()
          (const Idx_ i, const Idx_ j, const Idx_ k, const Idx_ l) const;

        ALIGNTYPE64       Tp_& JB_RESTRICT operator[] 
          (const Idx_ i);
        ALIGNTYPE64 const Tp_& JB_RESTRICT operator[] 
          (const Idx_ i) const;

        ALIGNTYPE64       Tp_* JB_RESTRICT data();
        ALIGNTYPE64 const Tp_* JB_RESTRICT data() const;

        void resize(const Idx_ sx, const Idx_ sy, const Idx_ sz, const Idx_ sw);

        const Idx_ sizeX() const;
        const Idx_ sizeY() const;
        const Idx_ sizeZ() const;
        const Idx_ sizeW() const;
        
        bool isDataAllocated() const;

        friend void swap(Array<Tp_,4,Idx_>& first, Array<Tp_,4,Idx_>& second) // nothrow
        { 
          std::swap(first.data_,second.data_);
          std::swap(first.sizeX_,second.sizeX_);
          std::swap(first.sizeY_,second.sizeY_);
          std::swap(first.sizeZ_,second.sizeZ_);
          std::swap(first.sizeW_,second.sizeW_);
        }



      private:

#ifdef BOUNDSCHECK
        bool isRangeValid(const Idx_ i, const Idx_ j, const Idx_ k, const Idx_ l) const;
        bool isRangeValid(const Idx_ i) const;

        std::string rangeErrorMessage(const Idx_ i, const Idx_ j, const Idx_ k, const Idx_ l) const;
        std::string rangeErrorMessage(const Idx_ i) const;
#endif

        Idx_ sizeX_;
        Idx_ sizeY_;
        Idx_ sizeZ_;
        Idx_ sizeW_;
        ALIGNTYPE64 Tp_* JB_RESTRICT  data_;
    };

///////////////////////////////////////////////////////////////////////
//
// implementation
//
///////////////////////////////////////////////////////////////////////


  template < typename Tp_, typename Idx_ >
    Array<Tp_,4,Idx_>::Array(const Idx_ sx, const Idx_ sy, const Idx_ sz, const Idx_ sw)
    : sizeX_(sx), sizeY_(sy), sizeZ_(sz), sizeW_(sw),
    data_( sizeX_*sizeY_*sizeZ_*sizeW_ ? (Tp_*)allocate_aligned64(sizeX_*sizeY_*sizeZ_*sizeW_*sizeof(data_)) : NULL ) 
    {}

  template < typename Tp_, typename Idx_ >
    Array<Tp_,4,Idx_>::Array(const Array<Tp_,4,Idx_ >& other)
    : sizeX_(other.sizeX_), sizeY_(other.sizeY_), sizeZ_(other.sizeZ_), sizeW_(other.sizeW_),
    data_( sizeX_*sizeY_*sizeZ_*sizeW_ ? (Tp_*)allocate_aligned64(sizeX_*sizeY_*sizeZ_*sizeW_*sizeof(data_)) : NULL ) {
      std::copy( other.data_, (other.data_ + sizeX_*sizeY_*sizeZ_*sizeW_), data_);
    }

  template < typename Tp_, typename Idx_ >
    Array<Tp_,4,Idx_>::Array(const Idx_ sx, const Idx_ sy, const Idx_ sz, const Idx_ sw, const Tp_ ival)
    : sizeX_(sx), sizeY_(sy), sizeZ_(sz), sizeW_(sw), 
    data_( sizeX_*sizeY_*sizeZ_ ? (Tp_*)allocate_aligned64(sizeX_*sizeY_*sizeZ_*sizeW_*sizeof(data_)) : NULL ) 
    {
      for ( JB_REGISTER Idx_ i=0; i != sizeX_*sizeY_*sizeZ_*sizeW_; ++i) {
        data_[i] = ival;
      }
    }


  template < typename Tp_, typename Idx_ >
    Array<Tp_,4,Idx_>::~Array() {
      if( isDataAllocated() ){ 
        free(data_);
        data_ = NULL;
      }
    }

  // access operators
  template < typename Tp_, typename Idx_ >
    JB_INLINE ALIGNTYPE64 Tp_& JB_RESTRICT Array<Tp_,4,Idx_>::operator()
    (const Idx_ i, const Idx_ j, const Idx_ k, const Idx_ l){ 
      assert( isDataAllocated() );
#ifdef BOUNDSCHECK
      if( !isRangeValid(i,j,k,l) ){ throw std::out_of_range( rangeErrorMessage(i,j,k,l) ); }
#endif
      return data_[ ((i*sizeY_+j)*sizeZ_ + k)*sizeW_ + l]; 
    }

  template < typename Tp_, typename Idx_ >
    JB_INLINE ALIGNTYPE64 const Tp_& JB_RESTRICT Array<Tp_,4,Idx_>::operator()
    (const Idx_ i, const Idx_ j, const Idx_ k, const Idx_ l) const{
      assert( isDataAllocated() );
#ifdef BOUNDSCHECK
      if( !isRangeValid(i,j,k,l) ){ throw std::out_of_range( rangeErrorMessage(i,j,k,l) ); }
#endif
      return data_[ ((i*sizeY_+j)*sizeZ_ + k)*sizeW_ + l]; 
    }

  template < typename Tp_, typename Idx_ >
    JB_INLINE ALIGNTYPE64 Tp_& JB_RESTRICT Array<Tp_,4,Idx_>::operator[]
    (const Idx_ i){ 
      assert( isDataAllocated() );
#ifdef BOUNDSCHECK
      if( !isRangeValid(i) ){ throw std::out_of_range( rangeErrorMessage(i) ); }
#endif
      return data_[i]; 
    }

  template < typename Tp_, typename Idx_ >
    JB_INLINE ALIGNTYPE64 const Tp_& JB_RESTRICT Array<Tp_,4,Idx_>::operator[]
    (const Idx_ i) const{
      assert( isDataAllocated() );
#ifdef BOUNDSCHECK
      if( !isRangeValid(i) ){ throw std::out_of_range( rangeErrorMessage(i) ); }
#endif
      return data_[i]; 
    }

  template < typename Tp_, typename Idx_ >
    ALIGNTYPE64 Tp_* JB_RESTRICT Array<Tp_,4,Idx_>::data() {
      assert( isDataAllocated() );
      return data_;
    }

  template < typename Tp_, typename Idx_ >
    ALIGNTYPE64 const Tp_* JB_RESTRICT Array<Tp_,4,Idx_>::data() const{
      assert( isDataAllocated() );
      return data_;
    }

template < typename Tp_, typename Idx_ >
void Array<Tp_,4,Idx_>::resize(const Idx_ sx, const Idx_ sy, const Idx_ sz, const Idx_ sw){

  Array< Tp_,4,Idx_> newArray(sx,sy,sz,sw);

  // copy the smaller array dimensions
  for( JB_REGISTER Idx_ i = 0, iend = std::min(sx,sizeX_); i != iend; ++i){
    for( JB_REGISTER Idx_ j = 0, jend = std::min(sy,sizeY_); j != jend; ++j){
      for( JB_REGISTER Idx_ k = 0, kend = std::min(sz,sizeZ_); k != kend; ++k){
        for( JB_REGISTER Idx_ l = 0, lend = std::min(sw,sizeW_); l != lend; ++l){
          newArray(i,j,k,l) = data_[ ((i*sizeY_+j)*sizeZ_ + k)*sizeW_ + l ];
        }
      }
    }
  }

  swap(*this, newArray);
}

template < typename Tp_, typename Idx_ >
Array<Tp_,4,Idx_>& Array<Tp_,4,Idx_>::operator=(Array<Tp_,4,Idx_> rhs){
  swap(*this, rhs);
  return *this;
}

template < typename Tp_, typename Idx_ >
JB_INLINE const Idx_ Array<Tp_,4,Idx_>::sizeX() const {
  return sizeX_;
}

template < typename Tp_, typename Idx_ >
JB_INLINE const Idx_ Array<Tp_,4,Idx_>::sizeY() const {
  return sizeY_;
}

template < typename Tp_, typename Idx_ >
JB_INLINE const Idx_ Array<Tp_,4,Idx_>::sizeZ() const {
  return sizeZ_;
}

template < typename Tp_, typename Idx_ >
JB_INLINE const Idx_ Array<Tp_,4,Idx_>::sizeW() const {
  return sizeW_;
}

#ifdef BOUNDSCHECK
template < typename Tp_, typename Idx_ >
JB_INLINE bool Array<Tp_,4,Idx_>::isRangeValid(const Idx_ i, const Idx_ j, const Idx_ k, const Idx_ l) const {
  return !( i >= sizeX_ || j >= sizeY_  || k >= sizeZ_ || l >= sizeW_ );
}
#endif

#ifdef BOUNDSCHECK
template < typename Tp_, typename Idx_ >
JB_INLINE bool Array<Tp_,4,Idx_>::isRangeValid(const Idx_ i) const {
  return ( i < sizeX_*sizeY_*sizeZ_*sizeW_ );
}
#endif


template < typename Tp_, typename Idx_ >
JB_INLINE bool Array<Tp_,4,Idx_>::isDataAllocated() const {
  return ( data_ != NULL );
}

#ifdef BOUNDSCHECK
template < typename Tp_, typename Idx_ >
std::string Array<Tp_,4,Idx_>::rangeErrorMessage(const Idx_ i, const Idx_ j, const Idx_ k, const Idx_ l) const {
  std::ostringstream message;
  message << "Array<3>::operator() ";
  message << "subscript: ( " << i << " , " << j << " , " << k << " , " << l << " ) "; 
  message << "range: ( " << sizeX_ << " , " << sizeY_ << " , " << sizeZ_ << " , " << sizeW_ <<" ) "; 
  return message.str();
}
#endif

#ifdef BOUNDSCHECK
template < typename Tp_, typename Idx_ >
std::string Array<Tp_,4,Idx_>::rangeErrorMessage(const Idx_ i) const {
  std::ostringstream message;
  message << "Array<3>::operator[] ";
  message << "subscript: [ " << i << " ] "; 
  message << "range: ( " << sizeX_*sizeY_*sizeZ_*sizeW_ << " ) "; 
  return message.str();
}
#endif
}

