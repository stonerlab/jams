#pragma once

#include <string>

#include "jams/helpers/consts.h"

namespace {

  const std::string config_basic_cpu(R"(
    solver : {
      module = "llg-heun-cpu";
      t_step = 1.0e-16;
      t_min  = 1.0e-16;
      t_max  = 1.0e-16;
    };

    physics : {
      temperature = 1.0;
    };
    )");

  const std::string config_basic_gpu(R"(
    solver : {
      module = "llg-heun-gpu";
      t_step = 1.0e-16;
      t_min  = 1.0e-16;
      t_max  = 1.0e-16;
    };

    physics : {
      temperature = 1.0;
    };
    )");

  const std::string config_unitcell_sc(R"(
    materials = (
      { name      = "Fe";
        moment    = 2.0;
        spin      = [1.0, 0.0, 0.0];
      }
    );

    unitcell : {
      parameter = 0.3e-9; 

      basis = (
        [ 1.0, 0.0, 0.0],
        [ 0.0, 1.0, 0.0],
        [ 0.0, 0.0, 1.0]);
      positions = (
        ("Fe", [0.0, 0.0, 0.0])
        );
    };
    )");

    const std::string config_unitcell_sc_2_atom(R"(
    materials = (
      { name      = "FeA";
        moment    = 2.0;
        spin      = [1.0, 0.0, 0.0];
      },
      { name      = "FeB";
        moment    = 2.0;
        spin      = [1.0, 0.0, 0.0];
      }
    );

    unitcell : {
      parameter = 0.3e-9; 

      basis = (
        [ 2.0, 0.0, 0.0],
        [ 0.0, 1.0, 0.0],
        [ 0.0, 0.0, 1.0]);
      positions = (
        ("FeA", [0.0, 0.0, 0.0]),
        ("FeB", [0.5, 0.0, 0.0])
        );
    };
    )");

    const std::string config_unitcell_bcc_2_atom(R"(
    materials = (
      { name      = "FeA";
        moment    = 2.0;
        spin      = [1.0, 0.0, 0.0];
      },
      { name      = "FeB";
        moment    = 1.0;
        spin      = [1.0, 0.0, 0.0];
      }
    );

    unitcell : {
      parameter = 0.3e-9; 

      basis = (
        [ 1.0, 0.0, 0.0],
        [ 0.0, 1.0, 0.0],
        [ 0.0, 0.0, 1.0]);
      positions = (
        ("FeA", [0.0, 0.0, 0.0]),
        ("FeB", [0.5, 0.5, 0.0])
        );
    };
    )");

  const std::string config_unitcell_sc_AFM(R"(
    materials = (
      { name      = "FeA";
        moment    = 2.0;
        spin      = [0.0, 0.0, 1.0];
        transform = ([1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]);
      },
      { name      = "FeB";
        moment    = 2.0;
        spin      = [0.0, 0.0, -1.0];
        transform = ([-1.0, 0.0, 0.0], [0.0, -1.0, 0.0], [0.0, 0.0, -1.0]);
      }
    );

    unitcell : {
      parameter = 0.3e-9; 

      basis = (
        [ 2.0, 0.0, 0.0],
        [ 0.0, 2.0, 0.0],
        [ 0.0, 0.0, 1.0]);
      positions = (
        ("FeA", [0.0, 0.0, 0.0]),
        ("FeA", [0.5, 0.5, 0.0]),
        ("FeB", [0.5, 0.0, 0.0]),
        ("FeB", [0.0, 0.5, 0.0])
        );
    };
    )");

const std::string config_lattice_1D(R"(
  lattice : {
    size     = [2000, 1, 1];
    periodic = [true, false, false];
  };
)");

const std::string config_lattice_2D(R"(

  lattice : {
    size     = [256, 256, 1];
    periodic = [true, true, false];
  };
)");

const std::string config_lattice_2D_128(R"(

  lattice : {
    size     = [128, 128, 1];
    periodic = [true, true, false];
  };
)");

const std::string config_dipole_bruteforce_64(R"(
  hamiltonians = (
    {
      module = "dipole-bruteforce";
      r_cutoff = 64.0;
    }
  );
)");

const std::string config_dipole_bruteforce_128(R"(
  hamiltonians = (
    {
      module = "dipole-bruteforce";
      r_cutoff = 128.0;
    }
  );
)");

const std::string config_dipole_bruteforce_1000(R"(
  hamiltonians = (
    {
      module = "dipole-bruteforce";
      r_cutoff = 1000.0;
    }
  );
)");

    const std::string config_dipole_neartree_64(R"(
  hamiltonians = (
    {
      module = "dipole-neartree";
      r_cutoff = 64.0;
    }
  );
)");

    const std::string config_dipole_neartree_128(R"(
  hamiltonians = (
    {
      module = "dipole-neartree";
      r_cutoff = 128.0;
    }
  );
)");

    const std::string config_dipole_neartree_1000(R"(
  hamiltonians = (
    {
      module = "dipole-bruteforce";
      r_cutoff = 1000.0;
    }
  );
)");

const std::string config_dipole_fft_128(R"(
  hamiltonians = (
    {
      module = "dipole-fft";
      r_cutoff = 128.0;
    }
  );
)");

const std::string config_dipole_fft_1000(R"(
  hamiltonians = (
    {
      module = "dipole-fft";
      r_cutoff = 1000.0;
    }
  );
)");

const std::string config_dipole_tensor_128(R"(
  hamiltonians = (
    {
      module = "dipole-tensor";
      r_cutoff = 128.0;
    }
  );
)");

const std::string config_dipole_tensor_1000(R"(
  hamiltonians = (
    {
      module = "dipole-tensor";
      r_cutoff = 1000.0;
    }
  );
)");

// -(0.5 * mu0 / (4 pi)) * (mus / a^3)^2
// mus = 2.0 muB; a = 0.3 nm
const double analytic_prefactor  = -23595.95647978379; // J/m^3
const double numeric_prefactor = kBohrMagneton / (0.3e-9 * 0.3e-9 * 0.3e-9);
}