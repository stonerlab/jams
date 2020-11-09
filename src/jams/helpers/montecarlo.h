#ifndef JAMS_CORE_MONTECARLO_H
#define JAMS_CORE_MONTECARLO_H

#include <random>

#include "jams/core/globals.h"
#include "jams/helpers/consts.h"
#include "random.h"

namespace jams {
    namespace montecarlo {

        enum class MonteCarloMoveType {
            REFLECTION,
            UNIFORM,
            ANGLE
        };

        class MonteCarloMove {
        public:
            virtual Vec3 operator()(const Vec3& spin) = 0;
        };

        class MonteCarloReflectionMove : public MonteCarloMove {
        public:
            inline Vec3 operator()(const Vec3& spin) {
              return -spin;
            }
        };

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

        template <class RNG>
        class MonteCarloAngleMove : public MonteCarloMove {
        public:
            MonteCarloAngleMove(RNG * gen, const double sigma) :
                    gen_(gen),
                    sigma_(sigma){}

            inline Vec3 operator()(const Vec3& spin) {
              return normalize(spin + uniform_random_sphere(*gen_) * sigma_);
            }

        private:
            double sigma_ = 0.5;
            RNG * gen_;
          };

        // Get spin 'i' from the global spin array as a Vec3
        inline Vec3 get_spin(int i) {
            return {globals::s(i,0), globals::s(i,1), globals::s(i,2)};
        }

        // Set spin 'i' in the global spin array to 'spin'
        inline void set_spin(const int &i, const Vec3& spin) {
          globals::s(i, 0) = spin[0];
          globals::s(i, 1) = spin[1];
          globals::s(i, 2) = spin[2];
        }


        // Returns true if a move should be accepted based on energy difference according to
        // A \exp(-\Delta E / \beta) where beta = 1/(k_B T) (or mu_B /(k_B T) in internal JAMS units)
        inline bool accept_on_probability(const double& deltaE, const double& temperature, const double prefactor = 1.0) {
          using std::min;
          static std::uniform_real_distribution<> uniform_distribution;

          double beta = (kBohrMagneton / kBoltzmann) / temperature;
          return uniform_distribution(jams::instance().random_generator()) < prefactor*exp(min(0.0, -deltaE * beta));
        }

        // generate a random spin index from the global generator
        inline int random_spin_index() {
          return jams::instance().random_generator()(globals::num_spins);
        }
    }
}

#endif  // JAMS_CORE_MONTECARLO_H