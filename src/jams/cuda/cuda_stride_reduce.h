// cuda_stride_reduce.h                                                          -*-C++-*-
#ifndef INCLUDED_JAMS_CUDA_STRIDE_REDUCE
#define INCLUDED_JAMS_CUDA_STRIDE_REDUCE
// see https://stackoverflow.com/questions/24850442/strided-reduction-by-cuda-thrust

#include <thrust/iterator/counting_iterator.h>
#include <thrust/iterator/transform_iterator.h>
#include <thrust/iterator/permutation_iterator.h>
#include <thrust/functional.h>
#include <thrust/device_ptr.h>
#include <thrust/reduce.h>
#include <thrust/device_vector.h>

template <typename Iterator>
class ThrustStridedRange
{
public:

    typedef typename thrust::iterator_difference<Iterator>::type difference_type;

    struct stride_functor : public thrust::unary_function<difference_type,difference_type>
    {
        difference_type stride;

        stride_functor(difference_type stride)
            : stride(stride) {}

        __host__ __device__
        difference_type operator()(const difference_type& i) const
        {
          return stride * i;
        }
    };

    typedef typename thrust::counting_iterator<difference_type>                   CountingIterator;
    typedef typename thrust::transform_iterator<stride_functor, CountingIterator> TransformIterator;
    typedef typename thrust::permutation_iterator<Iterator,TransformIterator>     PermutationIterator;

    // type of the strided_range iterator
    typedef PermutationIterator iterator;

    // construct strided_range for the range [first,last)
    ThrustStridedRange(Iterator first, Iterator last, difference_type stride)
        : first(first), last(last), stride(stride) {}

    iterator begin(void) const
    {
      return PermutationIterator(first, TransformIterator(CountingIterator(0), stride_functor(stride)));
    }

    iterator end(void) const
    {
      return begin() + ((last - first) + (stride - 1)) / stride;
    }

protected:
    Iterator first;
    Iterator last;
    difference_type stride;
};


template <typename T>
inline T cuda_stride_reduce_array(T* dev_ptr, const size_t size, const size_t stride, const size_t offset=0) {
  typedef typename thrust::device_vector<T>::iterator Iterator;

  ThrustStridedRange<Iterator> pos(
      thrust::device_ptr<T>(dev_ptr) + offset,
      thrust::device_ptr<T>(dev_ptr) + offset + size,
      stride);

  return thrust::reduce(pos.begin(), pos.end(), (T) 0, thrust::plus<T>());
}





#endif
// ----------------------------- END-OF-FILE ----------------------------------