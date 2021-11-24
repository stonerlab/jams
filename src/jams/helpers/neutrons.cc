#include "jams/helpers/neutrons.h"
#include "jams/core/globals.h"
#include "jams/core/lattice.h"

using namespace std;
using namespace libconfig;

namespace jams {
// Calculates the approximate neutron form factor at |q|
// Approximation and constants from International Tables for Crystallography: Vol. C (pp. 454–461).
    double form_factor(const Vec3 &q, const double &lattice_parameter, FormFactorG &g, FormFactorJ &j) {
      //(*) How is "norm(q)/(4.0*kPi*lattice_parameter)" is related to s=sin(theta)/lambda in the ref.?
      auto s_sq = pow2(norm(q) / (4.0 * kPi * lattice_parameter));
      auto ffq = 0.0;
      for (auto l : {0, 2, 4, 6}) {
        double p = (l == 0) ? 1.0 : s_sq; //if l=0 then insert 1.0 to p, otherwise s_sq
        ffq += g[l] * p *
               (j[l].A * exp(-j[l].a * s_sq) + j[l].B * exp(-j[l].b * s_sq) + j[l].C * exp(-j[l].c * s_sq) + j[l].D);

//        cout << "j_params[" << l << "].A = " << j[l].A << endl;
//        cout << "j_params[" << l << "].a = " << j[l].a << endl;
//        cout << "j_params[" << l << "].B = " << j[l].B << endl;
//        cout << "j_params[" << l << "].b = " << j[l].b << endl;
//        cout << "j_params[" << l << "].C = " << j[l].C << endl;
//        cout << "j_params[" << l << "].c = " << j[l].c << endl;
//        cout << "j_params[" << l << "].D = " << j[l].D << endl;
//        cout << "g_params[" << l << "] = " << g[l] << endl;
      }
      return 0.5 * ffq;
    }

    std::pair<vector<FormFactorG>, vector<FormFactorJ>> read_form_factor_settings(Setting &settings) {
      auto num_materials = lattice->num_materials();

      if (settings.getLength() != num_materials) {
        throw runtime_error("there must be one form factor per material\"");
      }

      vector<FormFactorG> g_params(num_materials);
      vector<FormFactorJ> j_params(num_materials);

      for (auto i = 0; i < settings.getLength(); ++i) {
        for (auto l : {0,2,4,6}) {
          j_params[i][l] = config_optional<FormFactorCoeff>(settings[i], "j" + to_string(l), j_params[i][l]);
        }
        g_params[i] = config_required<FormFactorG>(settings[i], "g");
      }

      return {g_params, j_params};
    }

}
