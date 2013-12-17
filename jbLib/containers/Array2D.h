#include <new>
#include <iostream>

template < typename type, typename index >
class Array < type, 2, index>
{
    friend std::ostream& operator<<(std::ostream& os, Array< type, 2 >& x){

        uint32 i,j;
        for( i=0; i<x.dim0; ++i){
            for( j=0; j<x.dim1; ++j){
                os << i << "\t" << j << "\t" << x.data0[i][j] << "\n";
            }
        }

        return os;
    }

    public:

    // default constructor
    Array(const index s0=0, const index s1=0)
        : dim0(s0), dim1(s1), 
        data0( dim0*dim1 ? (type**)allocate_aligned64(dim0*sizeof(*data0)) : 0 ), 
        data1( dim0*dim1 ? (type*)allocate_aligned64(dim0*dim1*sizeof(*data1)) : 0 ) 
    {
        assert( checkAllocation() );

        linkPointerArrays();
    }

    // copy constructor
    Array( const Array< type, 2, index >& other )
        : dim0(other.dim0), dim1(other.dim1), 
        data0( dim0*dim1 ? (type**)allocate_aligned64(dim0*sizeof(*data0)) : 0 ), 
        data1( dim0*dim1 ? (type*)allocate_aligned64(dim0*dim1*sizeof(*data1)) : 0 ) 

    {
        assert( checkAllocation() );
        
        linkPointerArrays();

        std::copy( other.data1, other.data1 + dim0*dim1, data1);
    }


    Array(const index s0, const index s1, const type ival)
        : dim0(s0), dim1(s1), 
        data0( dim0*dim1 ? (type**)allocate_aligned64(dim0*sizeof(*data0)) : 0 ), 
        data1( dim0*dim1 ? (type*)allocate_aligned64(dim0*dim1*sizeof(*data1)) : 0 ) 
    {
        assert( checkAllocation() );
        
        linkPointerArrays();

        JB_REGISTER index i;
        for( i=0; i<s0*s1; ++i){
            data1[i] = ival;
        }
    }

    ~Array(){
        if(data0 != NULL){ free(data0); }
        if(data1 != NULL){ free(data1); }
    }

    void resize(const index s0, const index s1){

        JB_REGISTER index i,j;

        Array< type, 2, index > newArray(s0,s1);
        
        // copy the smaller array dimensions
        for(i = 0; i < std::min(s0,dim0); ++i){
            for(j = 0; j < std::min(s1,dim1); ++j){
                newArray.data0[i][j] = data0[i][j];
            }
        }
        
        swap(*this, newArray);
    }

     //Normal Access Operations
    JB_FORCE_INLINE ALIGNTYPE64 type& JB_RESTRICT operator()(const index i, const index j){ 
        assert( checkAllocation() );
        assert( checkBounds(i,j) );
        return data0[i][j]; 
    }
    
    JB_FORCE_INLINE ALIGNTYPE64 const type& JB_RESTRICT operator()(const index i, const index j) const{ 
        assert( checkAllocation() );
        assert( checkBounds(i,j) );
        return data0[i][j];
    }

    // Raw pointer access to data array
    JB_FORCE_INLINE ALIGNTYPE64 type& JB_RESTRICT operator[](const index i) {
        assert( checkAllocation() );
        assert( i < dim0*dim1 );
        return data1[i];
    }

    JB_FORCE_INLINE ALIGNTYPE64 const type& operator[](const index i) const {
        assert( checkAllocation() );
        assert( i < dim0*dim1 );
        return data1[i];
    }

    JB_FORCE_INLINE ALIGNTYPE64 type* JB_RESTRICT ptr() {
        assert( checkAllocation() );
        return data1;
    }
    
    JB_FORCE_INLINE ALIGNTYPE64 const type* JB_RESTRICT ptr() const {
        assert( checkAllocation() );
        return data1;
    }

    JB_FORCE_INLINE const index  size0() const {
        assert( checkAllocation() );
        return dim0;
    }
    
    JB_FORCE_INLINE const index  size1() const {
        assert( checkAllocation() );
        return dim1;
    }
    
    friend void swap(Array< type, 2, index >& first, Array< type, 2, index >& second) // nothrow
    { 
        std::swap(first.data0,second.data0);
        std::swap(first.data1,second.data1);
        std::swap(first.dim0,second.dim0);
        std::swap(first.dim1,second.dim1);
    }

    Array< type, 2, index >& operator=(Array< type, 2, index > rhs){
        swap(*this, rhs);
        return *this;
    }


  private:

    JB_INLINE void linkPointerArrays(){
        // Use multiple arrays of pointers because the auto vectorizers seem to
        // prefer this to the implied mathematics
        JB_REGISTER index i,j;

        for(i=0; i<dim0; ++i){
            data0[i] = data1 + (i*dim1);
        }
    }

    JB_INLINE bool checkBounds(const index i, const index j) const{
        if( i < dim0 && j < dim1){
            return true;
        } else {
            return false;
        }
    }

    JB_INLINE bool checkAllocation() const{
        if( (data0 == NULL) || (data1 == NULL) ){
            return false;
        } else {
            return true;
        }
    }

    index                dim0;
    index                dim1;
    ALIGNTYPE64 type**  JB_RESTRICT  data0;
    ALIGNTYPE64 type*   JB_RESTRICT  data1;
};
