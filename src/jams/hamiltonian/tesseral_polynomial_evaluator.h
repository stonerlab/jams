#ifndef JAMS_TESSERAL_POLYNOMIAL_EVALUATOR_H
#define JAMS_TESSERAL_POLYNOMIAL_EVALUATOR_H

#include "jams/maths/tesseral_harmonics.h"

namespace jams::tesseral_polynomial {

JAMS_HOST_DEVICE inline int axial_coefficient_index_from_key(const int key)
{
    switch (key) {
    case jams::tesseral_key(2, 0): return 0;
    case jams::tesseral_key(4, 0): return 1;
    case jams::tesseral_key(6, 0): return 2;
    default: return -1;
    }
}

template <typename T>
JAMS_HOST_DEVICE inline void local_coordinates(
    const T sx_global,
    const T sy_global,
    const T sz_global,
    const T ux,
    const T uy,
    const T uz,
    const T vx,
    const T vy,
    const T vz,
    const T wx,
    const T wy,
    const T wz,
    T local_spin[3])
{
    local_spin[0] = sx_global * ux + sy_global * uy + sz_global * uz;
    local_spin[1] = sx_global * vx + sy_global * vy + sz_global * vz;
    local_spin[2] = sx_global * wx + sy_global * wy + sz_global * wz;
}

template <typename T>
JAMS_HOST_DEVICE inline T energy_from_local_terms(
    const int term_begin,
    const int term_end,
    const int* const keys,
    const T* const coefficients,
    const T sx,
    const T sy,
    const T sz)
{
    T energy = 0;
    for (int n = term_begin; n < term_end; ++n) {
        energy += coefficients[n] * tesseral_monic_polynomial_key_lookup(keys[n], sx, sy, sz);
    }
    return energy;
}

template <typename T>
JAMS_HOST_DEVICE inline T axial_energy_from_coefficients(
    const T c20,
    const T c40,
    const T c60,
    const T z)
{
    return c20 * jams::tesseral_monic_polynomial_l2_0(T{0}, T{0}, z)
        + c40 * jams::tesseral_monic_polynomial_l4_0(T{0}, T{0}, z)
        + c60 * jams::tesseral_monic_polynomial_l6_0(T{0}, T{0}, z);
}

template <typename T>
JAMS_HOST_DEVICE inline T axial_negative_gradient_from_coefficients(
    const T c20,
    const T c40,
    const T c60,
    const T z)
{
    T grad[3];
    T gradient_z = 0;

    jams::tesseral_monic_polynomial_l2_0_grad(T{0}, T{0}, z, grad);
    gradient_z += c20 * grad[2];
    jams::tesseral_monic_polynomial_l4_0_grad(T{0}, T{0}, z, grad);
    gradient_z += c40 * grad[2];
    jams::tesseral_monic_polynomial_l6_0_grad(T{0}, T{0}, z, grad);
    gradient_z += c60 * grad[2];

    return -gradient_z;
}

template <typename T>
JAMS_HOST_DEVICE inline void negative_gradient_from_local_terms(
    const int term_begin,
    const int term_end,
    const int* const keys,
    const T* const coefficients,
    const T sx,
    const T sy,
    const T sz,
    T local_field[3])
{
    local_field[0] = 0;
    local_field[1] = 0;
    local_field[2] = 0;

    for (int n = term_begin; n < term_end; ++n) {
        T grad[3];
        tesseral_monic_polynomial_grad_key_lookup(keys[n], sx, sy, sz, grad);
        const T coefficient = coefficients[n];
        local_field[0] -= coefficient * grad[0];
        local_field[1] -= coefficient * grad[1];
        local_field[2] -= coefficient * grad[2];
    }
}

template <typename T>
JAMS_HOST_DEVICE inline void local_vector_to_global(
    const T local_x,
    const T local_y,
    const T local_z,
    const T ux,
    const T uy,
    const T uz,
    const T vx,
    const T vy,
    const T vz,
    const T wx,
    const T wy,
    const T wz,
    T global_vector[3])
{
    global_vector[0] = local_x * ux + local_y * vx + local_z * wx;
    global_vector[1] = local_x * uy + local_y * vy + local_z * wy;
    global_vector[2] = local_x * uz + local_y * vz + local_z * wz;
}

template <typename T>
JAMS_HOST_DEVICE inline T energy_for_spin(
    const int spin_index,
    const T sx_global,
    const T sy_global,
    const T sz_global,
    const T* const u_axes,
    const T* const v_axes,
    const T* const w_axes,
    const int* const spin_pointer,
    const int* const keys,
    const T* const coefficients)
{
    const int base = 3 * spin_index;

    T local_spin[3];
    local_coordinates(
        sx_global, sy_global, sz_global,
        u_axes[base + 0], u_axes[base + 1], u_axes[base + 2],
        v_axes[base + 0], v_axes[base + 1], v_axes[base + 2],
        w_axes[base + 0], w_axes[base + 1], w_axes[base + 2],
        local_spin);

    return energy_from_local_terms(
        spin_pointer[spin_index],
        spin_pointer[spin_index + 1],
        keys,
        coefficients,
        local_spin[0],
        local_spin[1],
        local_spin[2]);
}

template <typename T>
JAMS_HOST_DEVICE inline T energy_for_spin_with_axial_terms(
    const int spin_index,
    const T sx_global,
    const T sy_global,
    const T sz_global,
    const T* const u_axes,
    const T* const v_axes,
    const T* const w_axes,
    const int* const spin_pointer,
    const int* const keys,
    const T* const coefficients,
    const T* const axial_coefficients)
{
    const int term_begin = spin_pointer[spin_index];
    const int term_end = spin_pointer[spin_index + 1];
    const int axial_base = 3 * spin_index;
    const T c20 = axial_coefficients[axial_base + 0];
    const T c40 = axial_coefficients[axial_base + 1];
    const T c60 = axial_coefficients[axial_base + 2];

    if (term_begin == term_end && c20 == T{0} && c40 == T{0} && c60 == T{0}) {
        return T{0};
    }

    const int base = 3 * spin_index;
    if (term_begin == term_end) {
        const T z = sx_global * w_axes[base + 0]
            + sy_global * w_axes[base + 1]
            + sz_global * w_axes[base + 2];
        return axial_energy_from_coefficients(c20, c40, c60, z);
    }

    T local_spin[3];
    local_coordinates(
        sx_global, sy_global, sz_global,
        u_axes[base + 0], u_axes[base + 1], u_axes[base + 2],
        v_axes[base + 0], v_axes[base + 1], v_axes[base + 2],
        w_axes[base + 0], w_axes[base + 1], w_axes[base + 2],
        local_spin);

    return axial_energy_from_coefficients(c20, c40, c60, local_spin[2])
        + energy_from_local_terms(
            term_begin,
            term_end,
            keys,
            coefficients,
            local_spin[0],
            local_spin[1],
            local_spin[2]);
}

template <typename T>
JAMS_HOST_DEVICE inline void field_for_spin(
    const int spin_index,
    const T sx_global,
    const T sy_global,
    const T sz_global,
    const T* const u_axes,
    const T* const v_axes,
    const T* const w_axes,
    const int* const spin_pointer,
    const int* const keys,
    const T* const coefficients,
    T field[3])
{
    const int base = 3 * spin_index;
    const T ux = u_axes[base + 0];
    const T uy = u_axes[base + 1];
    const T uz = u_axes[base + 2];
    const T vx = v_axes[base + 0];
    const T vy = v_axes[base + 1];
    const T vz = v_axes[base + 2];
    const T wx = w_axes[base + 0];
    const T wy = w_axes[base + 1];
    const T wz = w_axes[base + 2];

    T local_spin[3];
    local_coordinates(
        sx_global, sy_global, sz_global,
        ux, uy, uz,
        vx, vy, vz,
        wx, wy, wz,
        local_spin);

    T local_field[3];
    negative_gradient_from_local_terms(
        spin_pointer[spin_index],
        spin_pointer[spin_index + 1],
        keys,
        coefficients,
        local_spin[0],
        local_spin[1],
        local_spin[2],
        local_field);

    local_vector_to_global(
        local_field[0],
        local_field[1],
        local_field[2],
        ux, uy, uz,
        vx, vy, vz,
        wx, wy, wz,
        field);
}

template <typename T>
JAMS_HOST_DEVICE inline void field_for_spin_with_axial_terms(
    const int spin_index,
    const T sx_global,
    const T sy_global,
    const T sz_global,
    const T* const u_axes,
    const T* const v_axes,
    const T* const w_axes,
    const int* const spin_pointer,
    const int* const keys,
    const T* const coefficients,
    const T* const axial_coefficients,
    T field[3])
{
    const int term_begin = spin_pointer[spin_index];
    const int term_end = spin_pointer[spin_index + 1];
    const int axial_base = 3 * spin_index;
    const T c20 = axial_coefficients[axial_base + 0];
    const T c40 = axial_coefficients[axial_base + 1];
    const T c60 = axial_coefficients[axial_base + 2];

    if (term_begin == term_end && c20 == T{0} && c40 == T{0} && c60 == T{0}) {
        field[0] = T{0};
        field[1] = T{0};
        field[2] = T{0};
        return;
    }

    const int base = 3 * spin_index;
    const T wx = w_axes[base + 0];
    const T wy = w_axes[base + 1];
    const T wz = w_axes[base + 2];

    if (term_begin == term_end) {
        const T z = sx_global * wx + sy_global * wy + sz_global * wz;
        const T axial_field_z = axial_negative_gradient_from_coefficients(c20, c40, c60, z);
        field[0] = axial_field_z * wx;
        field[1] = axial_field_z * wy;
        field[2] = axial_field_z * wz;
        return;
    }

    const T ux = u_axes[base + 0];
    const T uy = u_axes[base + 1];
    const T uz = u_axes[base + 2];
    const T vx = v_axes[base + 0];
    const T vy = v_axes[base + 1];
    const T vz = v_axes[base + 2];

    T local_spin[3];
    local_coordinates(
        sx_global, sy_global, sz_global,
        ux, uy, uz,
        vx, vy, vz,
        wx, wy, wz,
        local_spin);

    T local_field[3];
    negative_gradient_from_local_terms(
        term_begin,
        term_end,
        keys,
        coefficients,
        local_spin[0],
        local_spin[1],
        local_spin[2],
        local_field);
    local_field[2] += axial_negative_gradient_from_coefficients(c20, c40, c60, local_spin[2]);

    local_vector_to_global(
        local_field[0],
        local_field[1],
        local_field[2],
        ux, uy, uz,
        vx, vy, vz,
        wx, wy, wz,
        field);
}

} // namespace jams::tesseral_polynomial

#endif // JAMS_TESSERAL_POLYNOMIAL_EVALUATOR_H
