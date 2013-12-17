#include <new>
#include <iostream>

template < typename type, typename index >
class Array < type, 1, index>
{
    friend std::ostream& operator<<(std::ostream& os, Array< type, 1 >& x){

        uint32 i;
        for( i=0; i<x.dim0; ++i){
            os << i << "\t" << x.data0[i] << "\n";
        }

        return os;
    }
    public:

    // default constructor
    Array(const index s0=0)
        : dim0(s0), 
        data0( dim0 ? (type*)allocate_aligned64(dim0*sizeof(*data0)) : 0 ) 
    { 
        assert( checkAllocation() ); 
    }

    // copy constructor
    Array( const Array< type, 1, index >& other )
        : dim0(other.dim0), 
        data0( dim0 ? (type*)allocate_aligned64(dim0*sizeof(*data0)) : 0 ) 

    {
        assert( checkAllocation() );
        std::copy( other.data0, other.data2 + dim0, data0);
    }


    Array(const index s0, const type ival)
        : dim0(s0), 
        data0( dim0 ? (type*)allocate_aligned64(dim0*sizeof(*data0)) : 0 ) 
    {
        assert( checkAllocation() );
        
        JB_REGISTER index i;
        for( i=0; i<dim0; ++i){
            data0[i] = ival;
        }
    }

    ~Array(){
        if(data0 != NULL){ free(data0); }
    }

    void resize(const index s0){

        JB_REGISTER index i;

        Array< type, 1, index > newArray(s0);
        
        // copy the smaller array dimensions
        for(i = 0; i < std::min(s0,dim0); ++i){
            newArray.data0[i] = data0[i];
        }
        
        swap(*this, newArray);
    }

     //Normal Access Operations
    JB_FORCE_INLINE ALIGNTYPE64 type& JB_RESTRICT operator()(const index i){ 
        assert( checkAllocation() );
        assert( checkBounds(i,j,k) );
        return data0[i]; 
    }
    
    JB_FORCE_INLINE ALIGNTYPE64 const type& JB_RESTRICT operator()(const index i) const{ 
        assert( checkAllocation() );
        assert( checkBounds(i,j,k) );
        return data0[i];
    }

    // Raw pointer access to data array
    JB_FORCE_INLINE ALIGNTYPE64 type& JB_RESTRICT operator[](const index i) {
        assert( checkAllocation() );
        assert( i < dim0 );
        return data0[i];
    }

    JB_FORCE_INLINE ALIGNTYPE64 const type& operator[](const index i) const {
        assert( checkAllocation() );
        assert( i < dim0 );
        return data0[i];
    }

    JB_FORCE_INLINE ALIGNTYPE64 type* JB_RESTRICT ptr() {
        assert( checkAllocation() );
        return data0;
    }
    
    JB_FORCE_INLINE ALIGNTYPE64 const type* JB_RESTRICT ptr() const {
        assert( checkAllocation() );
        return data0;
    }

    JB_FORCE_INLINE const index& JB_RESTRICT size0() const {
        assert( checkAllocation() );
        return dim0;
    }
    
    friend void swap(Array< type, 1, index >& first, Array< type, 1, index >& second) // nothrow
    { 
        std::swap(first.data0,second.data0);
        std::swap(first.dim0,second.dim0);
    }

    Array< type, 1, index >& operator=(Array< type, 1, index > rhs){
        swap(*this, rhs);
        return *this;
    }

    


  private:

    JB_INLINE bool checkBounds(const index i) const{
        if( i < dim0 ){
            return true;
        } else {
            return false;
        }
    }

    JB_INLINE bool checkAllocation() const{
        if( (data0 == NULL) ){
            return false;
        } else {
            return true;
        }
    }

    index                          dim0;
    ALIGNTYPE64 type* JB_RESTRICT  data0;
};

