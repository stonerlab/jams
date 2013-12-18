#include <new>
#include <iostream>

#ifdef BOUNDSCHECK
#include <stdexcept>
#include <sstream>
#endif

namespace jbLib {
  template <typename Tp_, typename Idx_>
    class Array <Tp_,5,Idx_> {
      public:

        Array(const Idx_ s0=0, const Idx_ s1=0, const Idx_ s2=0, const Idx_ s3=0, const Idx_ s4=0);
        Array(const Idx_ s0, const Idx_ s1, const Idx_ s2, const Idx_ s3, const Idx_ s4, const Tp_ ival);

        ~Array();
        Array(const Array<Tp_,5,Idx_>& other);
        Array<Tp_,5,Idx_>& operator=(Array<Tp_,5,Idx_> rhs);

        ALIGNTYPE64       Tp_& JB_RESTRICT operator()
          (const Idx_ i, const Idx_ j, const Idx_ k, const Idx_ l, const Idx_ m);
        ALIGNTYPE64 const Tp_& JB_RESTRICT operator()
          (const Idx_ i, const Idx_ j, const Idx_ k, const Idx_ l, const Idx_ m) const;

        ALIGNTYPE64       Tp_& JB_RESTRICT operator[] 
          (const Idx_ i);
        ALIGNTYPE64 const Tp_& JB_RESTRICT operator[] 
          (const Idx_ i) const;

        ALIGNTYPE64       Tp_* JB_RESTRICT data();
        ALIGNTYPE64 const Tp_* JB_RESTRICT data() const;

        void resize(const Idx_ s0, const Idx_ s1, const Idx_ s2, const Idx_ s3, const Idx_ s4);
    
        const Idx_ size( const Idx_ i) const;


        bool isDataAllocated() const;

        friend void swap(Array<Tp_,5,Idx_>& first, Array<Tp_,5,Idx_>& second) // nothrow
        { 
          std::swap(first.data_,second.data_);
          std::swap(first.size0_,second.size0_);
          std::swap(first.size1_,second.size1_);
          std::swap(first.size2_,second.size2_);
          std::swap(first.size3_,second.size3_);
          std::swap(first.size4_,second.size4_);
        }



      private:
        const Idx_ totalSize() const;

#ifdef BOUNDSCHECK
        bool isRangeValid(const Idx_ i, const Idx_ j, const Idx_ k, const Idx_ l, const Idx_ m) const;
        bool isRangeValid(const Idx_ i) const;

        std::string rangeErrorMessage(const Idx_ i, const Idx_ j, const Idx_ k, const Idx_ l, const Idx_ m) const;
        std::string rangeErrorMessage(const Idx_ i) const;
#endif

        Idx_ size0_;
        Idx_ size1_;
        Idx_ size2_;
        Idx_ size3_;
        Idx_ size4_;
        ALIGNTYPE64 Tp_* JB_RESTRICT  data_;
    };

///////////////////////////////////////////////////////////////////////
//
// implementation
//
///////////////////////////////////////////////////////////////////////
  
  template < typename Tp_, typename Idx_ >
    JB_INLINE const Idx_ Array<Tp_,5,Idx_>::totalSize() const {
      return size0_*size1_*size2_*size3_*size4_;
    }

  template < typename Tp_, typename Idx_ >
    JB_INLINE const Idx_ Array<Tp_,5,Idx_>::size(const Idx_ i) const {
      return (&size0_)[i];
    }

  template < typename Tp_, typename Idx_ >
    Array<Tp_,5,Idx_>::Array(const Idx_ s0, const Idx_ s1, const Idx_ s2, const Idx_ s3, const Idx_ s4)
    : size0_(s0), size1_(s1), size2_(s2), size3_(s3), size4_(s4),
    data_( totalSize() ? (Tp_*)allocate_aligned64(totalSize()*sizeof(data_)) : NULL ) 
    {}

  template < typename Tp_, typename Idx_ >
    Array<Tp_,5,Idx_>::Array(const Array<Tp_,5,Idx_ >& other)
    : size0_(other.size0_), size1_(other.size1_), size2_(other.size2_), size3_(other.size3_), size4_(other.size4_),
    data_( totalSize() ? (Tp_*)allocate_aligned64(totalSize()*sizeof(data_)) : NULL ) {
      std::copy( other.data_, (other.data_ + totalSize()), data_);
    }

  template < typename Tp_, typename Idx_ >
    Array<Tp_,5,Idx_>::Array(const Idx_ s0, const Idx_ s1, const Idx_ s2, const Idx_ s3, const Idx_ s4, const Tp_ ival)
    : size0_(s0), size1_(s1), size2_(s2), size3_(s3), 
    data_( totalSize() ? (Tp_*)allocate_aligned64(totalSize()*sizeof(data_)) : NULL ) 
    {
      for ( JB_REGISTER Idx_ i=0; i != totalSize(); ++i) {
        data_[i] = ival;
      }
    }


  template < typename Tp_, typename Idx_ >
    Array<Tp_,5,Idx_>::~Array() {
      if( isDataAllocated() ){ 
        free(data_);
        data_ = NULL;
      }
    }

  // access operators
  template < typename Tp_, typename Idx_ >
    JB_INLINE ALIGNTYPE64 Tp_& JB_RESTRICT Array<Tp_,5,Idx_>::operator()
    (const Idx_ i, const Idx_ j, const Idx_ k, const Idx_ l, const Idx_ m){ 
      assert( isDataAllocated() );
#ifdef BOUNDSCHECK
      if( !isRangeValid(i,j,k,l,m) ){ throw std::out_of_range( rangeErrorMessage(i,j,k,l,m) ); }
#endif
      return data_[ (((i*size1_+j)*size2_ + k)*size3_ + l)*size4_ + m]; 
    }

  template < typename Tp_, typename Idx_ >
    JB_INLINE ALIGNTYPE64 const Tp_& JB_RESTRICT Array<Tp_,5,Idx_>::operator()
    (const Idx_ i, const Idx_ j, const Idx_ k, const Idx_ l, const Idx_ m) const{
      assert( isDataAllocated() );
#ifdef BOUNDSCHECK
      if( !isRangeValid(i,j,k,l,m) ){ throw std::out_of_range( rangeErrorMessage(i,j,k,l,m) ); }
#endif
      return data_[ (((i*size1_+j)*size2_ + k)*size3_ + l)*size4_ + m]; 
    }

  template < typename Tp_, typename Idx_ >
    JB_INLINE ALIGNTYPE64 Tp_& JB_RESTRICT Array<Tp_,5,Idx_>::operator[]
    (const Idx_ i){ 
      assert( isDataAllocated() );
#ifdef BOUNDSCHECK
      if( !isRangeValid(i) ){ throw std::out_of_range( rangeErrorMessage(i) ); }
#endif
      return data_[i]; 
    }

  template < typename Tp_, typename Idx_ >
    JB_INLINE ALIGNTYPE64 const Tp_& JB_RESTRICT Array<Tp_,5,Idx_>::operator[]
    (const Idx_ i) const{
      assert( isDataAllocated() );
#ifdef BOUNDSCHECK
      if( !isRangeValid(i) ){ throw std::out_of_range( rangeErrorMessage(i) ); }
#endif
      return data_[i]; 
    }

  template < typename Tp_, typename Idx_ >
    ALIGNTYPE64 Tp_* JB_RESTRICT Array<Tp_,5,Idx_>::data() {
      assert( isDataAllocated() );
      return data_;
    }

  template < typename Tp_, typename Idx_ >
    ALIGNTYPE64 const Tp_* JB_RESTRICT Array<Tp_,5,Idx_>::data() const{
      assert( isDataAllocated() );
      return data_;
    }

template < typename Tp_, typename Idx_ >
void Array<Tp_,5,Idx_>::resize(const Idx_ s0, const Idx_ s1, const Idx_ s2, const Idx_ s3, const Idx_ s4){

  Array< Tp_,5,Idx_> newArray(s0,s1,s2,s3,s4);

  // copy the smaller array dimensions
  for( JB_REGISTER Idx_ i = 0, iend = std::min(s0,size0_); i != iend; ++i){
    for( JB_REGISTER Idx_ j = 0, jend = std::min(s1,size1_); j != jend; ++j){
      for( JB_REGISTER Idx_ k = 0, kend = std::min(s2,size2_); k != kend; ++k){
        for( JB_REGISTER Idx_ l = 0, lend = std::min(s3,size3_); l != lend; ++l){
          for( JB_REGISTER Idx_ m = 0, mend = std::min(s4,size4_); m != mend; ++m){
            newArray(i,j,k,l,m) = data_[ (((i*size1_+j)*size2_ + k)*size3_ + l)*size4_ + m ];
          }
        }
      }
    }
  }

  swap(*this, newArray);
}

template < typename Tp_, typename Idx_ >
Array<Tp_,5,Idx_>& Array<Tp_,5,Idx_>::operator=(Array<Tp_,5,Idx_> rhs){
  swap(*this, rhs);
  return *this;
}

#ifdef BOUNDSCHECK
template < typename Tp_, typename Idx_ >
JB_INLINE bool Array<Tp_,5,Idx_>::isRangeValid(const Idx_ i, const Idx_ j, const Idx_ k, const Idx_ l, const Idx_ m) const {
  return !( i >= size0_ || j >= size1_  || k >= size2_ || l >= size3_ || m >= size4_ );
}
#endif

#ifdef BOUNDSCHECK
template < typename Tp_, typename Idx_ >
JB_INLINE bool Array<Tp_,5,Idx_>::isRangeValid(const Idx_ i) const {
  return ( i < totalSize() );
}
#endif


template < typename Tp_, typename Idx_ >
JB_INLINE bool Array<Tp_,5,Idx_>::isDataAllocated() const {
  return ( data_ != NULL );
}

#ifdef BOUNDSCHECK
template < typename Tp_, typename Idx_ >
std::string Array<Tp_,5,Idx_>::rangeErrorMessage(const Idx_ i, const Idx_ j, const Idx_ k, const Idx_ l, const Idx_ m) const {
  std::ostringstream message;
  message << "Array<3>::operator() ";
  message << "subscript: ( " << i << " , " << j << " , " << k << " , " << l << " , " << m << " ) "; 
  message << "range: ( " << size0_ << " , " << size1_ << " , " << size2_ << " , " << size3_ << " , " << size4_ <<" ) "; 
  return message.str();
}
#endif

#ifdef BOUNDSCHECK
template < typename Tp_, typename Idx_ >
std::string Array<Tp_,5,Idx_>::rangeErrorMessage(const Idx_ i) const {
  std::ostringstream message;
  message << "Array<3>::operator[] ";
  message << "subscript: [ " << i << " ] "; 
  message << "range: ( " << totalSize() << " ) "; 
  return message.str();
}
#endif
}

