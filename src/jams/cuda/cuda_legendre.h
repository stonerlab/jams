//
// Created by Joseph Barker on 08/01/2026.
//

#ifndef JAMS_CUDA_LEGENDRE_H
#define JAMS_CUDA_LEGENDRE_H

#include <type_traits>

template <typename T>
__device__ inline T cuda_legendre_poly_0(T x) {
    static_assert(std::is_floating_point<T>::value,
                  "cuda_legendre_poly requires floating-point type");
    (void)x;
    return T(1);
}

template <typename T>
__device__ inline T cuda_legendre_poly_1(T x) {
    static_assert(std::is_floating_point<T>::value, "");
    return x;
}

template <typename T>
__device__ inline T cuda_legendre_poly_2(T x) {
    static_assert(std::is_floating_point<T>::value, "");
    return T(1.5) * x * x - T(0.5);
}

template <typename T>
__device__ inline T cuda_legendre_poly_3(T x) {
    static_assert(std::is_floating_point<T>::value, "");
    return T(2.5) * x * x * x - T(1.5) * x;
}

template <typename T>
__device__ inline T cuda_legendre_poly_4(T x) {
    static_assert(std::is_floating_point<T>::value, "");
    return T(4.375) * x * x * x * x
         - T(3.75)  * x * x
         + T(0.375);
}

template <typename T>
__device__ inline T cuda_legendre_poly_5(T x) {
    static_assert(std::is_floating_point<T>::value, "");
    return T(7.875) * x * x * x * x * x
         - T(8.75)  * x * x * x
         + T(1.875) * x;
}

template <typename T>
__device__ inline T cuda_legendre_poly_6(T x) {
    static_assert(std::is_floating_point<T>::value, "");
    return T(14.4375) * x * x * x * x * x * x
         - T(19.6875) * x * x * x * x
         + T(6.5625)  * x * x
         - T(0.3125);
}

template <typename T>
__device__ inline T cuda_legendre_dpoly_0(T x) {
    static_assert(std::is_floating_point<T>::value, "");
    (void)x;
    return T(0);
}

template <typename T>
__device__ inline T cuda_legendre_dpoly_1(T x) {
    static_assert(std::is_floating_point<T>::value, "");
    (void)x;
    return T(1);
}

template <typename T>
__device__ inline T cuda_legendre_dpoly_2(T x) {
    static_assert(std::is_floating_point<T>::value, "");
    return T(3) * x;
}

template <typename T>
__device__ inline T cuda_legendre_dpoly_3(T x) {
    static_assert(std::is_floating_point<T>::value, "");
    return T(7.5) * x * x - T(1.5);
}

template <typename T>
__device__ inline T cuda_legendre_dpoly_4(T x) {
    static_assert(std::is_floating_point<T>::value, "");
    return T(17.5) * x * x * x - T(7.5) * x;
}

template <typename T>
__device__ inline T cuda_legendre_dpoly_5(T x) {
    static_assert(std::is_floating_point<T>::value, "");
    return T(39.375) * x * x * x * x
         - T(26.25)  * x * x
         + T(1.875);
}

template <typename T>
__device__ inline T cuda_legendre_dpoly_6(T x) {
    static_assert(std::is_floating_point<T>::value, "");
    return T(86.625) * x * x * x * x * x
         - T(78.75)  * x * x * x
         + T(13.125) * x;
}

template <typename T>
__device__ inline T cuda_legendre_poly(T x, int n) {
    static_assert(std::is_floating_point<T>::value, "");

    switch (n) {
    case 0: return cuda_legendre_poly_0(x);
    case 1: return cuda_legendre_poly_1(x);
    case 2: return cuda_legendre_poly_2(x);
    case 3: return cuda_legendre_poly_3(x);
    case 4: return cuda_legendre_poly_4(x);
    case 5: return cuda_legendre_poly_5(x);
    case 6: return cuda_legendre_poly_6(x);
    }

    T pn1 = cuda_legendre_poly_2(x);
    T pn2 = cuda_legendre_poly_1(x);
    T pn  = pn1;

    for (int l = 3; l <= n; ++l) {
        pn = ( (T(2) * l - T(1)) * x * pn1
             - (T(l - 1) * pn2) ) / T(l);
        pn2 = pn1;
        pn1 = pn;
    }

    return pn;
}

template <typename T>
__device__ inline T cuda_legendre_dpoly(T x, int n) {
    static_assert(std::is_floating_point<T>::value, "");

    switch (n) {
    case 0: return cuda_legendre_dpoly_0(x);
    case 1: return cuda_legendre_dpoly_1(x);
    case 2: return cuda_legendre_dpoly_2(x);
    case 3: return cuda_legendre_dpoly_3(x);
    case 4: return cuda_legendre_dpoly_4(x);
    case 5: return cuda_legendre_dpoly_5(x);
    case 6: return cuda_legendre_dpoly_6(x);
    }

    return (x * cuda_legendre_poly(x, n)
          - cuda_legendre_poly(x, n - 1))
         / T(2 * n + 1);
}


#endif //JAMS_CUDA_LEGENDRE_H