#ifndef JAMS_CORE_MONTECARLO_H
#define JAMS_CORE_MONTECARLO_H

#include <random>

#include "jams/core/globals.h"
#include "jams/helpers/consts.h"
#include "random.h"

namespace jams {
    namespace montecarlo {

        /// Pure virtual base class describing the interface for Monte Carlo
        /// moves. The class works as a functor which accepts a spin (Vec3) and
        /// returns an altered (moved) spin (Vec3) which will be used as a trial
        /// change in a Monte Carlo solver. Complicated behaviour can be
        /// achieved through using a constructor and private data members.
        class MonteCarloMove {
        public:
            virtual Vec3 operator()(const Vec3& spin) = 0;
        };

        /// Implementation of MonteCarloMove which reflects a spin
        /// \f[
        /// (S_x, S_y, S_z) \rightarrow (-S_x, -S_y, -S_z)
        /// \f]
        /// @WARNING This move is not ergodic, it cannot sample all of the
        /// possible configuration space and MUST be used alongside an ergodic
        ///
        /// @details This move is provided mainly to enable rapid equilibration
        /// in systems which have a non-collinear or (or anti parallel) ground
        /// state.
        class MonteCarloReflectionMove : public MonteCarloMove {
        public:
            inline Vec3 operator()(const Vec3& spin) {
              return -spin;
            }
        };

        /// Implementation of MonteCarloMove which produces a randomly oriented
        /// spin. The move is independent of the input spin.
        ///
        /// @details The constructor requires a pointer to a random generator.
        /// This move is always safe to use but may have a low
        /// acceptance rate once thermal equilibrium is reached.
        template <class RNG>
        class MonteCarloUniformMove : public MonteCarloMove {
        public:
            explicit MonteCarloUniformMove(RNG * gen) :
                    gen_(gen)
                    {}

            inline Vec3 operator()(const Vec3& spin) {
              return uniform_random_sphere(*gen_);
            }

        private:
            RNG * gen_;
        };

        /// Implementation of MonteCarloMove which moves randomly moves the spin
        /// within a defined opening angle
        /// \f[
        /// \mathbf{S} \rightarrow \mathbf{S} + sigma \mathbf{\Omega}
        /// \f]
        /// where \f$\Omega\f$ is a random vector on the unit sphere.
        ///
        /// @details The constructor requires a pointer to a random generator
        /// and a value for sigma (default = 0.5).
        /// This move is usually optimal for thermodynamic sampling.
        /// While sigma does not preciscley relate to an opening angle it allows
        /// much faster computation than dealing explicitly with angles. Tuning
        /// the value of sigma allows simple tuning of the acceptance rate in
        /// Monte Carlo. Close to thermal equilibrium small moves are more
        /// likely to be accepted, but it will take more moves to sample
        /// configuration space.
        template <class RNG>
        class MonteCarloAngleMove : public MonteCarloMove {
        public:
            MonteCarloAngleMove(RNG * gen, const double sigma) :
                    gen_(gen),
                    sigma_(sigma){}

            inline Vec3 operator()(const Vec3& spin) {
              return normalize(fma(sigma_, uniform_random_sphere(*gen_), spin));
            }

        private:
            const double sigma_ = 0.5;
            RNG * gen_;
          };

        /// Get spin 'i' from the global spin array as a Vec3
        inline Vec3 get_spin(int i) {
            return {globals::s(i,0), globals::s(i,1), globals::s(i,2)};
        }

        /// Set spin 'i' in the global spin array to 'spin'
        inline void set_spin(const int &i, const Vec3& spin) {
          globals::s(i, 0) = spin[0];
          globals::s(i, 1) = spin[1];
          globals::s(i, 2) = spin[2];
        }


        /// Returns true if a Monte Carlo move should be accepted based on
        /// energy difference according to the Boltzmann distribution
        /// \f$ A \exp(-\Delta E / \beta) \f$ where \f$\beta = 1/(k_B T)\f$
        /// (or \f$mu_B /(k_B T)\f$ in internal JAMS units). The prefactor
        /// \f$ A \f$ allows the distribution to be modified as used in some
        /// solvers such as constrained Monte Carlo.
        inline bool accept_on_boltzmann_distribution(const double& deltaE, const double& temperature, const double prefactor = 1.0) {
          using std::min;
          static std::uniform_real_distribution<> uniform_distribution;

          double beta = (kBohrMagnetonIU / kBoltzmannIU) / temperature;
          return uniform_distribution(jams::instance().random_generator()) < prefactor*exp(min(0.0, -deltaE * beta));
        }

        /// Generate a random spin index from the global generator
        inline int random_spin_index() {
          return jams::instance().random_generator()(globals::num_spins);
        }
    }
}

#endif  // JAMS_CORE_MONTECARLO_H