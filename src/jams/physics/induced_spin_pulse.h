// induced_spin_pulse.h                                                -*-C++-*-
#ifndef INCLUDED_JAMS_PHYSICS_INDUCED_SPIN_PULSE
#define INCLUDED_JAMS_PHYSICS_INDUCED_SPIN_PULSE

/// @class InducedSpinPulsePhysics
///
/// Increases the length of spins with a gaussian temporal profile.
///
/// @details
/// @p This class is to simulate physics where the length of atomic moments is
/// changed by laser excitations. For example the OISTR process
/// [Dewhurst, Nano Lett. 18, 1842 (2018)](https://dx.doi.org/10.1021/acs.nanolett.7b05118).
/// It does not implement a detailed physical model nor does it transfer spin
/// moment between sites. It simply adjust the spin lengths.
///
/// There are two modes, the incoherent mode adjusts the length of spins along
/// their current direction
/// @f[
///    \vec{S}_i(t) = \vec{S}_i(t) + A\frac{\vec{S}_i(t)}{|\vec{S}_i(t)|}\exp\left(-\frac{(t-t_0)^2}{2\sigma^2} \right)
/// @f]
/// where A is the pulse height, t_0 is the pulse center time and sigma is the
/// pulse width.
///
/// The second mode adds spin moment coherently in a specified polarisation
/// direction
///
/// @f[
///    \vec{S}_i(t) = \vec{S}_i(t)+ A\vec{p}\exp\left(-\frac{(t-t_0)^2}{2\sigma^2} \right)
/// @f]
/// where \vec{p} is a unit vector along the polarisation direction.
///
/// Note that the gaussian is scaling the moments which are being
/// integrated in time, so the pulse height is not the maximum size of the
/// adjustment. The pulse height generally wants to be very small, order of
/// 1e-5.
///
/// By default the pulse is applied to all spins. The pulse can be restricted to
/// a single material using the `material` setting.
///
/// @attention In JAMS this scales the dimensionless spin vectors, not the
/// magnetic moments (in Bohr magnetons). Therefore this should only be used
/// with solvers which allow non-unit vector spins (e.g. the GSE solver). For
/// solvers which assume or enforce unit vector spins the use of this Physics
/// module may give undefined behaviour.
///
/// @setting `physics.pulse_center_time` (required) center time of the Gaussian pulse in units of picoseconds
/// @setting `physics.pulse_width` (required) temporal width of the Gaussian pulse in units of picoseconds
/// @setting `physics.pulse_height` (required) height of Gaussian pulse in dimensionless units
/// @setting `physics.pulse_is_coherent` (required) toggles whether to use the coherent or incoherent mode
/// @setting `physics.pulse_polarisation` (required if pulse_is_coherent = true) 3 vector defining the pulse polarisation
/// @setting `physics.material` (optional) restrict the application of the pulse to the specified material
/// @example Example config:
/// @code{.unparsed}
/// physics: {
///     module = "induced-spin-pulse";
///     pulse_center_time = 10.0;
///     pulse_width = 0.5;
///     pulse_height = 1e-5;
///     pulse_is_coherent = true;
///     pulse_polarisation = [0.0, 0.0, 1.0];
///     material = "Rh";
///     temperature = 10.0;
/// };
/// @endcode

#include <jams/core/physics.h>
#include <jams/interface/config.h>
#include <jams/containers/multiarray.h>

class InducedSpinPulsePhysics : public Physics {
public:
    explicit InducedSpinPulsePhysics(const libconfig::Setting &settings);
    ~InducedSpinPulsePhysics() override = default;
    void update(const int &iterations, const double &time, const double &dt) override;
private:
    double pulse_center_time_;
    double pulse_width_;
    double pulse_height_;
    bool   pulse_is_coherent_;
    Vec3   pulse_polarisation_;

    jams::MultiArray<int, 1> spin_indices_;
};

#endif  // INCLUDED_JAMS_PHYSICS_INDUCED_SPIN_PULSE
// ----------------------------- END-OF-FILE ----------------------------------