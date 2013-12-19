#include <new>
#include <iostream>
#include "../sys/intrinsics.h"

#ifdef BOUNDSCHECK
#include <stdexcept>
#include <sstream>
#endif

#include "../sys/intrinsics.h"

namespace jblib {
  template <typename Tp_, typename Idx_>
    class Array <Tp_,1,Idx_> {
      public:

        Array(const Idx_ sx=0);
        Array(const Idx_ sx, const Tp_ ival);

        ~Array();
        Array(const Array<Tp_,1,Idx_>& other);
        Array<Tp_,1,Idx_>& operator=(Array<Tp_,1,Idx_> rhs);

        ALIGNTYPE64       Tp_& JB_RESTRICT operator()
          (const Idx_ i);
        ALIGNTYPE64 const Tp_& JB_RESTRICT operator()
          (const Idx_ i) const;

        ALIGNTYPE64       Tp_& JB_RESTRICT operator[] 
          (const Idx_ i);
        ALIGNTYPE64 const Tp_& JB_RESTRICT operator[] 
          (const Idx_ i) const;

        ALIGNTYPE64       Tp_* JB_RESTRICT data();
        ALIGNTYPE64 const Tp_* JB_RESTRICT data() const;

        void resize(const Idx_ sx);

        const Idx_ size( ) const;

        
        bool isDataAllocated() const;

        friend void swap(Array<Tp_,1,Idx_>& first, Array<Tp_,1,Idx_>& second) // nothrow
        { 
          std::swap(first.data_,second.data_);
          std::swap(first.sizeX_,second.sizeX_);
        }

        const Idx_ totalSize() const;

      private:

#ifdef BOUNDSCHECK
        bool isRangeValid(const Idx_ i) const;

        std::string rangeErrorMessage(const Idx_ i) const;
#endif

        Idx_ sizeX_;
        ALIGNTYPE64 Tp_* JB_RESTRICT  data_;
    };

///////////////////////////////////////////////////////////////////////
//
// implementation
//
///////////////////////////////////////////////////////////////////////

  template < typename Tp_, typename Idx_ >
    JB_INLINE const Idx_ Array<Tp_,1,Idx_>::totalSize() const {
      return sizeX_;
    }
  
  template < typename Tp_, typename Idx_ >
    JB_INLINE const Idx_ Array<Tp_,1,Idx_>::size() const {
      return sizeX_;
    }


  template < typename Tp_, typename Idx_ >
    Array<Tp_,1,Idx_>::Array(const Idx_ sx)
    : sizeX_(sx), 
    data_( sizeX_ ? (Tp_*)allocate_aligned64(sizeX_*sizeof(data_)) : NULL ) 
    {}

  template < typename Tp_, typename Idx_ >
    Array<Tp_,1,Idx_>::Array(const Array<Tp_,1,Idx_ >& other)
    : sizeX_(other.sizeX_), 
    data_( sizeX_ ? (Tp_*)allocate_aligned64(sizeX_*sizeof(data_)) : NULL ) {
      std::copy( other.data_, (other.data_ + sizeX_), data_);
    }

  template < typename Tp_, typename Idx_ >
    Array<Tp_,1,Idx_>::Array(const Idx_ sx, const Tp_ ival)
    : sizeX_(sx), 
    data_( sizeX_ ? (Tp_*)allocate_aligned64(sizeX_*sizeof(data_)) : NULL ) 
    {
      for ( JB_REGISTER Idx_ i=0; i != sizeX_; ++i) {
        data_[i] = ival;
      }
    }


  template < typename Tp_, typename Idx_ >
    Array<Tp_,1,Idx_>::~Array() {
      if( isDataAllocated() ){ 
        free(data_);
        data_ = NULL;
      }
    }

  // access operators
  template < typename Tp_, typename Idx_ >
    JB_INLINE ALIGNTYPE64 Tp_& JB_RESTRICT Array<Tp_,1,Idx_>::operator()
    (const Idx_ i){ 
      assert( isDataAllocated() );
#ifdef BOUNDSCHECK
      if( !isRangeValid(i) ){ throw std::out_of_range( rangeErrorMessage(i) ); }
#endif
      return data_[i]; 
    }

  template < typename Tp_, typename Idx_ >
    JB_INLINE ALIGNTYPE64 const Tp_& JB_RESTRICT Array<Tp_,1,Idx_>::operator()
    (const Idx_ i) const{
      assert( isDataAllocated() );
#ifdef BOUNDSCHECK
      if( !isRangeValid(i) ){ throw std::out_of_range( rangeErrorMessage(i) ); }
#endif
      return data_[i]; 
    }

  template < typename Tp_, typename Idx_ >
    JB_INLINE ALIGNTYPE64 Tp_& JB_RESTRICT Array<Tp_,1,Idx_>::operator[]
    (const Idx_ i){ 
      assert( isDataAllocated() );
#ifdef BOUNDSCHECK
      if( !isRangeValid(i) ){ throw std::out_of_range( rangeErrorMessage(i) ); }
#endif
      return data_[i]; 
    }

  template < typename Tp_, typename Idx_ >
    JB_INLINE ALIGNTYPE64 const Tp_& JB_RESTRICT Array<Tp_,1,Idx_>::operator[]
    (const Idx_ i) const{
      assert( isDataAllocated() );
#ifdef BOUNDSCHECK
      if( !isRangeValid(i) ){ throw std::out_of_range( rangeErrorMessage(i) ); }
#endif
      return data_[i]; 
    }

  template < typename Tp_, typename Idx_ >
    ALIGNTYPE64 Tp_* JB_RESTRICT Array<Tp_,1,Idx_>::data() {
      assert( isDataAllocated() );
      return data_;
    }

  template < typename Tp_, typename Idx_ >
    ALIGNTYPE64 const Tp_* JB_RESTRICT Array<Tp_,1,Idx_>::data() const{
      assert( isDataAllocated() );
      return data_;
    }

  template < typename Tp_, typename Idx_ >
    void Array<Tp_,1,Idx_>::resize(const Idx_ sx){

      Array< Tp_,1,Idx_> newArray(sx);

      // copy the smaller array dimensions
      for( JB_REGISTER Idx_ i = 0, iend = std::min(sx,sizeX_); i != iend; ++i){
        newArray(i) = data_[i];
      }

      swap(*this, newArray);
    }

  template < typename Tp_, typename Idx_ >
    Array<Tp_,1,Idx_>& Array<Tp_,1,Idx_>::operator=(Array<Tp_,1,Idx_> rhs){
      swap(*this, rhs);
      return *this;
    }


#ifdef BOUNDSCHECK
  template < typename Tp_, typename Idx_ >
    JB_INLINE bool Array<Tp_,1,Idx_>::isRangeValid(const Idx_ i) const {
      return ( i < sizeX_ );
    }
#endif


  template < typename Tp_, typename Idx_ >
    JB_INLINE bool Array<Tp_,1,Idx_>::isDataAllocated() const {
      return ( data_ != NULL );
    }

#ifdef BOUNDSCHECK
  template < typename Tp_, typename Idx_ >
    std::string Array<Tp_,1,Idx_>::rangeErrorMessage(const Idx_ i) const {
      std::ostringstream message;
      message << "Array<1>::operator() ";
      message << "subscript: ( " << i <<  " ) "; 
      message << "range: ( " << sizeX_ << " ) "; 
      return message.str();
    }
#endif

}

