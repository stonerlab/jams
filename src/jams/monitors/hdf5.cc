// Copyright 2014 Joseph Barker. All rights reserved.

#include <cstdio>
#include <climits>
#include <string>
#include <algorithm>

#include "version.h"

#include "jams/core/physics.h"
#include "jams/core/solver.h"
#include "jams/core/globals.h"
#include "jams/core/lattice.h"
#include "jams/helpers/utils.h"
#include "jams/helpers/slice.h"
#include "jams/interface/config.h"
#include "jams/interface/highfive.h"
#include "jams/helpers/output.h"

#include "jams/monitors/hdf5.h"

namespace {
    const int h5_compression_chunk_size = 4095;
    const int h5_compression_factor = 6;

    template <typename T>
    constexpr unsigned xdmf_precision() {
      return static_cast<unsigned>(sizeof(T));
    }

    HighFive::DataSetCreateProps dataset_create_props(
        const bool compression_enabled,
        std::initializer_list<hsize_t> chunk_dims) {
      HighFive::DataSetCreateProps props;

      if (compression_enabled) {
        props.add(HighFive::Chunking(chunk_dims));
        props.add(HighFive::Shuffle());
        props.add(HighFive::Deflate(h5_compression_factor));
      }

      return props;
    }

    template <typename T>
    void write_vector_field(
        const jams::MultiArray<T, 2>& field,
        const std::string& data_path,
        HighFive::File& file,
        const bool compression_enabled) {
      const auto chunk_rows = static_cast<hsize_t>(
          std::min(h5_compression_chunk_size, int(field.extent(0))));
      auto props = dataset_create_props(compression_enabled, {chunk_rows, 1});
      auto dataset = file.createDataSet<T>(
          data_path,
          HighFive::DataSpace({size_t(field.extent(0)), size_t(field.extent(1))}),
          props);
      dataset.write(field);
    }

    template <typename T>
    void write_scalar_field(
        const jams::MultiArray<T, 1>& field,
        const std::string& data_path,
        HighFive::File& file,
        const bool compression_enabled) {
      const auto chunk_size = static_cast<hsize_t>(
          std::min(h5_compression_chunk_size, int(field.size())));
      auto props = dataset_create_props(compression_enabled, {chunk_size});
      auto dataset = file.createDataSet<T>(
          data_path,
          HighFive::DataSpace({size_t(field.size())}),
          props);
      dataset.write(field);
    }
}

Hdf5Monitor::Hdf5Monitor(const libconfig::Setting &settings)
: Monitor(settings),
  slice_() {
    output_step_freq_ = settings["output_steps"];

    // it outsteps is 0 then use max int instead - output will only be generated in the
    // constructor and destructor
    if (output_step_freq_ == 0){
      output_step_freq_ = INT_MAX;
    }

    // compression options
    compression_enabled_ = jams::config_optional<bool>(settings, "compressed", compression_enabled_);
  std::cout << "  compressed " << compression_enabled_ << "\n";

    if (settings.exists("slice")) {
        slice_ = Slice(settings["slice"]);
    }

    write_ds_dt_ = jams::config_optional<bool>(settings, "ds_dt", write_ds_dt_);

    open_new_xdmf_file(jams::output::monitor_filename(name(), "xdmf"));

    write_lattice_h5_file(jams::output::monitor_filename(name() + "_lattice", "h5"));
}

Hdf5Monitor::~Hdf5Monitor() {
  // always write final in double precision
    const auto final_h5_file = jams::output::monitor_filename(name() + "_final", "h5");
    write_spin_h5_file(final_h5_file);
    update_xdmf_file(final_h5_file, globals::solver->time());

    fclose(xdmf_file_);
}



void Hdf5Monitor::update(Solver& solver) {
  if (solver.iteration()%output_step_freq_ == 0) {
    int outcount = solver.iteration()/output_step_freq_;  // int divisible by modulo above

    const std::string h5_file_name(jams::output::monitor_filename_series(name(), "h5", outcount));

    write_spin_h5_file(h5_file_name);
    update_xdmf_file(h5_file_name, solver.time());
  }
}

void Hdf5Monitor::write_spin_h5_file(const std::string &h5_file_name) {
  using namespace HighFive;

  File file(h5_file_name, File::ReadWrite | File::Create | File::Truncate);

  if (slice_.num_points() != 0) {
    const auto point_count = output_point_count();
    jams::MultiArray<double, 2> spins(point_count, 3);
    const auto spin_view = globals::s.host_view();
    auto output_view = spins.mutable_host_view();

    for (auto output_index = 0; output_index < point_count; ++output_index) {
      const auto spin = source_spin_index(output_index);
      for (auto component = 0; component < 3; ++component) {
        output_view(output_index, component) = spin_view(spin, component);
      }
    }

    write_vector_field(spins, "/spins", file, compression_enabled_);
  } else {
    write_vector_field(globals::s, "/spins", file, compression_enabled_);
  }

  if (write_ds_dt_) {
    if (slice_.num_points() != 0) {
      const auto point_count = output_point_count();
      jams::MultiArray<double, 2> ds_dt(point_count, 3);
      const auto ds_dt_view = globals::ds_dt.host_view();
      auto output_view = ds_dt.mutable_host_view();

      for (auto output_index = 0; output_index < point_count; ++output_index) {
        const auto spin = source_spin_index(output_index);
        for (auto component = 0; component < 3; ++component) {
          output_view(output_index, component) = ds_dt_view(spin, component);
        }
      }

      write_vector_field(ds_dt, "/ds_dt", file, compression_enabled_);
    } else {
      write_vector_field(globals::ds_dt, "/ds_dt", file, compression_enabled_);
    }
  }
}

//---------------------------------------------------------------------

void Hdf5Monitor::write_lattice_h5_file(const std::string &h5_file_name) {
  using namespace HighFive;

  File file(h5_file_name, File::ReadWrite | File::Create | File::Truncate);

  const auto point_count = output_point_count();
  jams::MultiArray<int, 1> types(point_count);
  jams::MultiArray<jams::Real, 1> moments(point_count);
  jams::MultiArray<jams::Real, 1> alpha(point_count);
  jams::MultiArray<jams::Real, 1> temperature(point_count);
  jams::MultiArray<double, 2> positions(point_count, 3);

  const auto moments_view = globals::mus.host_view();
  const auto alpha_view = globals::alpha.host_view();

  const auto* thermostat = globals::solver != nullptr ? globals::solver->thermostat() : nullptr;
  const auto* physics = globals::solver != nullptr ? globals::solver->physics() : nullptr;
  const auto* temperature_profile = thermostat != nullptr ? &thermostat->temperature_profile() : nullptr;
  const bool has_per_spin_temperature =
      temperature_profile != nullptr && temperature_profile->is_per_spin();
  const auto temperature_view = has_per_spin_temperature
      ? temperature_profile->temperature().host_view()
      : jams::MultiArray<jams::Real, 1>::const_host_view_type{};

  auto types_output = types.mutable_host_view();
  auto moments_output = moments.mutable_host_view();
  auto alpha_output = alpha.mutable_host_view();
  auto temperature_output = temperature.mutable_host_view();
  auto positions_output = positions.mutable_host_view();

  jams::Real uniform_temperature = 0.0;
  if (temperature_profile != nullptr && temperature_profile->is_uniform()) {
    uniform_temperature = temperature_profile->uniform_temperature();
  } else if (physics != nullptr) {
    uniform_temperature = static_cast<jams::Real>(physics->temperature());
  } else if (globals::config != nullptr && globals::config->exists("physics.temperature")) {
    uniform_temperature = static_cast<jams::Real>(
        jams::config_required<double>(globals::config->lookup("physics"), "temperature"));
  }

  for (auto output_index = 0; output_index < point_count; ++output_index) {
    const auto spin = source_spin_index(output_index);
    types_output(output_index) = globals::lattice->lattice_site_material_id(spin);
    moments_output(output_index) = moments_view(spin);
    alpha_output(output_index) = alpha_view(spin);
    temperature_output(output_index) = has_per_spin_temperature
        ? temperature_view(spin)
        : uniform_temperature;

    const auto position = globals::lattice->lattice_site_position_cart(spin);
    for (auto component = 0; component < 3; ++component) {
      positions_output(output_index, component) =
          globals::lattice->parameter() * position[component] / 1e-9;
    }
  }

  auto type_dataset = file.createDataSet<int>("/types",  DataSpace({size_t(point_count)}));
  type_dataset.write(types);
  write_scalar_field(moments, "/moments", file, compression_enabled_);
  write_scalar_field(alpha, "/alpha", file, compression_enabled_);
  write_scalar_field(temperature, "/temperature", file, compression_enabled_);
  auto pos_dataset = file.createDataSet<double>("/positions",  DataSpace({size_t(point_count), 3}));
  pos_dataset.createAttribute<std::string>("units", DataSpace::From("nm"));
  pos_dataset.write(positions);

}

//---------------------------------------------------------------------

void Hdf5Monitor::open_new_xdmf_file(const std::string &xdmf_file_name) {
  // create xdmf_file_
  xdmf_file_ = fopen(xdmf_file_name.c_str(), "w");

               fputs("<?xml version=\"1.0\"?>\n", xdmf_file_);
               fputs("<!DOCTYPE Xdmf SYSTEM \"https://gitlab.kitware.com/xdmf/xdmf/raw/master/Xdmf.dtd\"[]>\n", xdmf_file_);
               fputs("<Xdmf Version=\"3.0\">\n", xdmf_file_);
               fputs("  <Domain Name=\"JAMS\">\n", xdmf_file_);
  fprintf(xdmf_file_, "    <Information Name=\"Commit\" Value=\"%s\" />\n", jams::build::hash);
  fprintf(xdmf_file_, "    <Information Name=\"Configuration\" Value=\"%s\" />\n", globals::simulation_name.c_str());
               fputs("    <Grid Name=\"TimeSeries\" GridType=\"Collection\" CollectionType=\"Temporal\">\n", xdmf_file_);
               fputs("    </Grid>\n", xdmf_file_);
               fputs("  </Domain>\n", xdmf_file_);
               fputs("</Xdmf>", xdmf_file_);
               fflush(xdmf_file_);
}

//---------------------------------------------------------------------

int Hdf5Monitor::output_point_count() {
  if (slice_.num_points() != 0) {
    return slice_.num_points();
  }

  return globals::num_spins;
}

int Hdf5Monitor::source_spin_index(const int output_index) {
  if (slice_.num_points() != 0) {
    return slice_.index(output_index);
  }

  return output_index;
}

//---------------------------------------------------------------------

void Hdf5Monitor::write_xdmf_scalar_attribute(
    const std::string& name,
    const std::string& h5_file_name,
    const std::string& data_path,
    const unsigned data_dimension,
    const unsigned precision,
    const std::string& number_type) {
  fprintf(xdmf_file_,
          "       <Attribute Name=\"%s\" AttributeType=\"Scalar\" Center=\"Node\">\n",
          name.c_str());
  fprintf(xdmf_file_,
          "         <DataItem Dimensions=\"%u\" NumberType=\"%s\" Precision=\"%u\" Format=\"HDF\">\n",
          data_dimension,
          number_type.c_str(),
          precision);
  fprintf(xdmf_file_,
          "           %s:%s\n",
          file_basename(h5_file_name).c_str(),
          data_path.c_str());
  fputs("         </DataItem>\n", xdmf_file_);
  fputs("       </Attribute>\n", xdmf_file_);
}

void Hdf5Monitor::write_xdmf_vector_attribute(
    const std::string& name,
    const std::string& h5_file_name,
    const std::string& data_path,
    const unsigned data_dimension,
    const unsigned precision) {
  fprintf(xdmf_file_,
          "       <Attribute Name=\"%s\" AttributeType=\"Vector\" Center=\"Node\">\n",
          name.c_str());
  fprintf(xdmf_file_,
          "         <DataItem Dimensions=\"%u 3\" NumberType=\"Float\" Precision=\"%u\" Format=\"HDF\">\n",
          data_dimension,
          precision);
  fprintf(xdmf_file_,
          "           %s:%s\n",
          file_basename(h5_file_name).c_str(),
          data_path.c_str());
  fputs("         </DataItem>\n", xdmf_file_);
  fputs("       </Attribute>\n", xdmf_file_);
}

//---------------------------------------------------------------------

void Hdf5Monitor::update_xdmf_file(const std::string &h5_file_name, const double time) {
  const auto data_dimension = static_cast<unsigned>(output_point_count());
  const auto lattice_h5_file = jams::output::monitor_filename(name() + "_lattice", "h5");

               // rewind the closing tags of the XML  (Grid, Domain, Xdmf)
               fseek(xdmf_file_, -31, SEEK_CUR);

  fprintf(xdmf_file_, "      <Grid Name=\"Lattice\" GridType=\"Uniform\">\n");
  fprintf(xdmf_file_, "        <Time Value=\"%f\" />\n", time);
  fprintf(xdmf_file_, "        <Topology TopologyType=\"Polyvertex\" Dimensions=\"%u\" />\n", data_dimension);
               fputs("       <Geometry GeometryType=\"XYZ\">\n", xdmf_file_);
  fprintf(xdmf_file_, "         <DataItem Dimensions=\"%u 3\" NumberType=\"Float\" Precision=\"%u\" Format=\"HDF\">\n", data_dimension, xdmf_precision<double>());
  fprintf(xdmf_file_, "           %s:/positions\n",
          file_basename(lattice_h5_file).c_str());
               fputs("         </DataItem>\n", xdmf_file_);
               fputs("       </Geometry>\n", xdmf_file_);
  write_xdmf_scalar_attribute("Type", lattice_h5_file, "/types", data_dimension, 4, "Int");
  write_xdmf_scalar_attribute("Moment", lattice_h5_file, "/moments", data_dimension, xdmf_precision<jams::Real>(), "Float");
  write_xdmf_scalar_attribute("Alpha", lattice_h5_file, "/alpha", data_dimension, xdmf_precision<jams::Real>(), "Float");
  write_xdmf_scalar_attribute("Temperature", lattice_h5_file, "/temperature", data_dimension, xdmf_precision<jams::Real>(), "Float");
  write_xdmf_vector_attribute("spin", h5_file_name, "/spins", data_dimension, xdmf_precision<double>());
  if (write_ds_dt_) {
    write_xdmf_vector_attribute("ds_dt", h5_file_name, "/ds_dt", data_dimension, xdmf_precision<double>());
  }
               fputs("      </Grid>\n", xdmf_file_);
               // reprint the closing tags of the XML
               fputs("    </Grid>\n", xdmf_file_);
               fputs("  </Domain>\n", xdmf_file_);
               fputs("</Xdmf>", xdmf_file_);
  fflush(xdmf_file_);
}
