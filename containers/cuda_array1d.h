#include <new>
#include <iostream> 
#include <stdexcept>

#include <cuda.h>
#include <cuda_runtime.h>

#include "array.h"

namespace jblib {
  template <typename Tp_, typename Idx_>
    class CudaArray <Tp_,1,Idx_> {
      public:

        CudaArray(const Idx_ sx=0);
        CudaArray(const Idx_ sx, const Tp_ ival);

        ~CudaArray();

        // Construct a CudaArray from a host Array
        template <uint32 HostDim_>
        CudaArray(const Array<Tp_,HostDim_,Idx_>& hostArray);
        
        // Construct a CudaArray from a device CudaArray
        CudaArray(const CudaArray<Tp_,1,Idx_>& devCudaArray);

              Tp_* data();
        const Tp_* data() const;

        void resize(const Idx_ sx);

        const Idx_ size() const;
        
        bool isDataAllocated() const;

        friend void swap(CudaArray<Tp_,1,Idx_>& first, CudaArray<Tp_,1,Idx_>& second) // nothrow
        { 
          std::swap(first.data_,second.data_);
          std::swap(first.sizeX_,second.sizeX_);
        }

        // DON'T OVERLOAD '=' OPERATOR
        // This would hide an expensive task. Instead implement as functions
        
        template <uint32 HostDim_>
        void copyFromHostArray(Array<Tp_,HostDim_,Idx_> &hostArray);

        template <uint32 HostDim_>
        void copyToHostArray(Array<Tp_,HostDim_,Idx_> &hostArray);

        CudaArray<Tp_,1,Idx_>& operator=(CudaArray<Tp_,1,Idx_> rhs);

      private:

        Idx_ sizeX_;
        Tp_* data_;
    };

///////////////////////////////////////////////////////////////////////
//
// implementation
//
///////////////////////////////////////////////////////////////////////


  template < typename Tp_, typename Idx_ >
    CudaArray<Tp_,1,Idx_>::CudaArray(const Idx_ sx)
    : sizeX_(sx), data_(NULL)
    {
      if (sizeX_ != 0) {
        std::cout << cudaMalloc( (void**)&data_,sizeX_*sizeof(data_)) << std::endl;
        if (cudaMalloc( (void**)&data_,sizeX_*sizeof(data_)) != cudaSuccess ) {
          
          throw std::bad_alloc();
        }
      }
    }
  
  template < typename Tp_, typename Idx_ >
    CudaArray<Tp_,1,Idx_>::~CudaArray() {
      if ( isDataAllocated() ) {
        if (cudaFree(data_) != cudaSuccess) {
          throw std::runtime_error("cudaFree fail");
        }
      }
    }



  template < typename Tp_, typename Idx_>
    template<uint32 HostDim_>
    CudaArray<Tp_,1,Idx_>::CudaArray(const Array<Tp_,HostDim_,Idx_>& hostArray) {

      sizeX_ = hostArray.totalSize();

      if (sizeX_ != 0) {
        if (cudaMalloc( (void**)&data_,sizeX_*sizeof(data_)) != cudaSuccess ) {
          throw std::bad_alloc();
        }
      }

      copyFromHostArray(hostArray);

    }


  template < typename Tp_, typename Idx_>
    template <uint32 HostDim_>
  void CudaArray<Tp_,1,Idx_>::copyFromHostArray(Array<Tp_,HostDim_,Idx_> &hostArray){

    if (sizeX_ != hostArray.totalSize()) {
      throw std::runtime_error("copyFromHostArray fail - size mismatch");
    }

    if (cudaMemcpy(data_,hostArray.data(),(size_t)(sizeX_*sizeof(data_)),cudaMemcpyHostToDevice) != cudaSuccess) {
      throw std::runtime_error("copyFromHostArray fail - cudaMemcpy failure.");
    }

  }

  template < typename Tp_, typename Idx_>
    template <uint32 HostDim_>
  void CudaArray<Tp_,1,Idx_>::copyToHostArray(Array<Tp_,HostDim_,Idx_> &hostArray){

    if (sizeX_ != hostArray.totalSize()) {
      throw std::runtime_error("copyToHostArray fail - size mismatch");
    }

    if (cudaMemcpy(hostArray.data(),data_,(size_t)(sizeX_*sizeof(data_)),cudaMemcpyDeviceToHost) != cudaSuccess) {
      throw std::runtime_error("copyToHostArray fail - cudaMemcpy failure.");
    }

  }

  template < typename Tp_, typename Idx_ >
    Tp_* CudaArray<Tp_,1,Idx_>::data() {
      assert( isDataAllocated() );
      return data_;
    }

  template < typename Tp_, typename Idx_ >
    const Tp_* CudaArray<Tp_,1,Idx_>::data() const{
      assert( isDataAllocated() );
      return data_;
    }

  template < typename Tp_, typename Idx_ >
    void CudaArray<Tp_,1,Idx_>::resize(const Idx_ sx){

      CudaArray< Tp_,1,Idx_> newCudaArray(sx);

      // copy the smaller array dimensions
      if (sizeX_ != 0 ) {
        if (cudaMemcpy(newCudaArray.data_,data_,(size_t)(std::min(sx,sizeX_)*sizeof(data_)),cudaMemcpyDeviceToDevice) != cudaSuccess) {
          throw std::runtime_error("resize fail - cudaMemcpy failure.");
        }
      }

      swap(*this, newCudaArray);
    }

  template < typename Tp_, typename Idx_ >
    CudaArray<Tp_,1,Idx_>& CudaArray<Tp_,1,Idx_>::operator=(CudaArray<Tp_,1,Idx_> rhs){
      swap(*this, rhs);
      return *this;
    }

  template < typename Tp_, typename Idx_ >
    JB_INLINE const Idx_ CudaArray<Tp_,1,Idx_>::size() const {
      return sizeX_;
    }


  template < typename Tp_, typename Idx_ >
    JB_INLINE bool CudaArray<Tp_,1,Idx_>::isDataAllocated() const {
      return ( data_ != NULL );
    }


}
