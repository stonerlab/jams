#include <new>
#include <iostream>

// consider checking if the products s0*s1*s2 would overflow the chosen index
// type
//
template < typename type, typename index >
class Array < type, 3, index>
{
    friend std::ostream& operator<<(std::ostream& os, Array< type, 3 >& x){

        uint32 i,j,k;
        for( i=0; i<x.dim0; ++i){
            for( j=0; j<x.dim1; ++j){
                for( k=0; k<x.dim2; ++k){
                    os << i << "\t" << j << "\t" << k << "\t" << x.data0[i][j][k] << "\n";
                }
            }
        }

        return os;
    }

    public:

    // default constructor
    Array(const index s0=0, const index s1=0, const index s2=0)
        : dim0(s0), dim1(s1), dim2(s2), 
        data0( dim0*dim1*dim2 ? (type***)allocate_aligned64(dim0*sizeof(*data0)) : 0 ), 
        data1( dim0*dim1*dim2 ? (type**)allocate_aligned64(dim0*dim1*sizeof(*data1)) : 0 ), 
        data2( dim0*dim1*dim2 ? (type*)allocate_aligned64(dim0*dim1*dim2*sizeof(*data2)) : 0 ) 
    {
        assert( checkAllocation() );

        linkPointerArrays();
    }

    // copy constructor
    Array( const Array< type, 3, index >& other )
        : dim0(other.dim0), dim1(other.dim1), dim2(other.dim2), 
        data0( dim0*dim1*dim2 ? (type***)allocate_aligned64(dim0*sizeof(*data0)) : 0 ), 
        data1( dim0*dim1*dim2 ? (type**)allocate_aligned64(dim0*dim1*sizeof(*data1)) : 0 ), 
        data2( dim0*dim1*dim2 ? (type*)allocate_aligned64(dim0*dim1*dim2*sizeof(*data2)) : 0 ) 

    {
        assert( checkAllocation() );
        
        linkPointerArrays();

        std::copy( other.data2, other.data2 + dim0*dim1*dim2, data2);
    }


    Array(const index s0, const index s1, const index s2, const type ival)
        : dim0(s0), dim1(s1), dim2(s2), 
        data0( dim0*dim1*dim2 ? (type***)allocate_aligned64(dim0*sizeof(*data0)) : 0 ), 
        data1( dim0*dim1*dim2 ? (type**)allocate_aligned64(dim0*dim1*sizeof(*data1)) : 0 ), 
        data2( dim0*dim1*dim2 ? (type*)allocate_aligned64(dim0*dim1*dim2*sizeof(*data2)) : 0 ) 
    {
        assert( checkAllocation() );
        
        linkPointerArrays();

         JB_REGISTER index i;
        for( i=0; i<s0*s1*s2; ++i){
            data2[i] = ival;
        }
    }

    ~Array(){
        if(data0 != NULL){ free(data0); }
        if(data1 != NULL){ free(data1); }
        if(data2 != NULL){ free(data2); }
    }

    void resize(const index s0, const index s1, const index s2){

        JB_REGISTER index i,j,k;

        Array< type, 3, index > newArray(s0,s1,s2);
        
        // copy the smaller array dimensions
        for(i = 0; i < std::min(s0,dim0); ++i){
            for(j = 0; j < std::min(s1,dim1); ++j){
                for(k = 0; k < std::min(s2,dim2); ++k){
                    newArray.data0[i][j][k] = data0[i][j][k];
                }
            }
        }
        
        swap(*this, newArray);
    }

     //Normal Access Operations
    JB_FORCE_INLINE ALIGNTYPE64 type& JB_RESTRICT operator()(const index i, const index j, const index k){ 
        assert( checkAllocation() );
        assert( checkBounds(i,j,k) );
        return data0[i][j][k]; 
    }
    
    JB_FORCE_INLINE ALIGNTYPE64 const type& JB_RESTRICT operator()(const index i, const index j, const index k) const{ 
        assert( checkAllocation() );
        assert( checkBounds(i,j,k) );
        return data0[i][j][k];
    }

    // Raw pointer access to data array
    JB_FORCE_INLINE ALIGNTYPE64 type& JB_RESTRICT operator[](const index i) {
        assert( checkAllocation() );
        assert( i < dim0*dim1*dim2 );
        return data2[i];
    }

    JB_FORCE_INLINE ALIGNTYPE64 const type& operator[](const index i) const {
        assert( checkAllocation() );
        assert( i < dim0*dim1*dim2 );
        return data2[i];
    }

    JB_FORCE_INLINE ALIGNTYPE64 type* JB_RESTRICT ptr() {
        assert( checkAllocation() );
        return data2;
    }
    
    JB_FORCE_INLINE ALIGNTYPE64 const type* JB_RESTRICT ptr() const {
        assert( checkAllocation() );
        return data2;
    }

    JB_FORCE_INLINE const index  size0() const {
        assert( checkAllocation() );
        return dim0;
    }
    
    JB_FORCE_INLINE const index  size1() const {
        assert( checkAllocation() );
        return dim1;
    }
    
    JB_FORCE_INLINE const index  size2() const {
        assert( checkAllocation() );
        return dim2;
    }

    friend void swap(Array< type, 3, index >& first, Array< type, 3, index >& second) // nothrow
    { 
        std::swap(first.data0,second.data0);
        std::swap(first.data1,second.data1);
        std::swap(first.data2,second.data2);
        std::swap(first.dim0,second.dim0);
        std::swap(first.dim1,second.dim1);
        std::swap(first.dim2,second.dim2);
    }

    Array< type, 3, index >& operator=(Array< type, 3, index > rhs){
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
            for(j=0; j<dim1; ++j){
                data1[i*dim1 + j] = data2 + (i*dim1+j)*dim2;
            }
        }
    }

    JB_INLINE bool checkBounds(const index& i, const index& j, const index& k) const{
        if( i < dim0 && j < dim1 && k < dim2){
            return true;
        } else {
            return false;
        }
    }

    JB_INLINE bool checkAllocation() const{
        if( (data0 == NULL) || (data1 == NULL) || (data2 == NULL) ){
            return false;
        } else {
            return true;
        }
    }

    index                dim0;
    index                dim1;
    index                dim2;
    ALIGNTYPE64 type***  JB_RESTRICT  data0;
    ALIGNTYPE64 type**   JB_RESTRICT  data1;
    ALIGNTYPE64 type*    JB_RESTRICT  data2;
};
