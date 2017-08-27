// Copyright 2014 Joseph Barker. All rights reserved.

#include <cstdint>
#include <string>

#include "jams/core/error.h"
#include "jams/core/output.h"
#include "jams/core/physics.h"
#include "jams/core/solver.h"
#include "jams/core/globals.h"
#include "jams/core/lattice.h"
#include "jams/core/utils.h"
#include "jams/monitors/vtu.h"

#include "jblib/math/equalities.h"

#define QUOTEME_(x) #x
#define QUOTEME(x) QUOTEME_(x)

VtuMonitor::VtuMonitor(const libconfig::Setting &settings)
: Monitor(settings) {
    using namespace globals;
    using jblib::floats_are_greater_than_or_equal;
    using jblib::floats_are_less_than_or_equal;

    ::output->write("\nInitialising Vtu monitor...\n");

    // settings for only outputting a slice
    if (settings.exists("slice_origin") ^ settings.exists("slice_size")) {
        jams_error("Xyz monitor requires both slice_origin and slice_size to be specificed;");
    }

    num_slice_points = 0;

    if (settings.exists("slice_origin")) {
        ::output->write("  slice output enabled\n");
        for (int i = 0; i < 3; ++i) {
            slice_origin[i] = settings["slice_origin"][i];
        }
        ::output->write("  slice origin: %f %f %f\n", slice_origin[0], slice_origin[1], slice_origin[2]);
        for (int i = 0; i < 3; ++i) {
            slice_size[i] = settings["slice_size"][i];
        }
        ::output->write("  slice size: %f %f %f\n", slice_size[0], slice_size[1], slice_size[2]);

        // check which spins are inside the slice
        for (int i = 0; i < num_spins; ++i) {
            Vec3 pos = ::lattice->atom_position(i);

            // check if the current spin in inside the slice
            if (floats_are_greater_than_or_equal(pos[0], slice_origin[0]) && floats_are_less_than_or_equal(pos[0], slice_origin[0] + slice_size[0])
            &&  floats_are_greater_than_or_equal(pos[1], slice_origin[1]) && floats_are_less_than_or_equal(pos[1], slice_origin[1] + slice_size[1])
            &&  floats_are_greater_than_or_equal(pos[2], slice_origin[2]) && floats_are_less_than_or_equal(pos[2], slice_origin[2] + slice_size[2])) {
              slice_spins.push_back(i);
            }
        }

        num_slice_points = slice_spins.size();

        types_binary_data.resize(num_slice_points);
        points_binary_data.resize(num_slice_points, 3);
        spins_binary_data.resize(num_slice_points, 3);

        for (int i = 0; i < num_slice_points; ++i) {
            types_binary_data(i) = ::lattice->atom_material(slice_spins[i]);
            for (int j = 0; j < 3; ++j) {
                points_binary_data(i, j) = ::lattice->parameter()*::lattice->atom_position(slice_spins[i])[j];
            }
        }
    } else {

        points_binary_data.resize(num_spins, 3);
        types_binary_data.resize(num_spins);

        for (int i = 0; i < num_spins; ++i) {
            types_binary_data(i) = ::lattice->atom_material(i);
            for (int j = 0; j < 3; ++j) {
                points_binary_data(i, j) = ::lattice->parameter()*::lattice->atom_position(i)[j];
            }
        }
    }

    output_step_freq_ = settings["output_steps"];
}

void VtuMonitor::update(Solver * solver) {
  using namespace globals;

  if (solver->iteration()%output_step_freq_ == 0) {
    int outcount = solver->iteration()/output_step_freq_;  // int divisible by modulo above

    std::ofstream vtkfile(std::string(seedname+"_"+zero_pad_number(outcount)+".vtu").c_str());


    uint32_t header_bytesize, types_bytesize, points_bytesize, spins_bytesize, num_points;

    if (num_slice_points == 0) {
        num_points = num_spins;
        header_bytesize = sizeof(uint32_t);
        types_bytesize  = num_spins*sizeof(int32_t);
        points_bytesize = num_spins3*sizeof(float);
        spins_bytesize  = num_spins3*sizeof(double);
    } else {
        num_points = num_slice_points;
        header_bytesize = sizeof(uint32_t);
        types_bytesize  = num_slice_points*sizeof(int32_t);
        points_bytesize = 3*num_slice_points*sizeof(float);
        spins_bytesize  = 3*num_slice_points*sizeof(double);
        for (int i = 0; i < num_slice_points; ++i) {
            for (int j = 0; j < 3; ++j) {
                spins_binary_data(i,j) = s(slice_spins[i],j);
            }
        }
    }

    vtkfile << "<?xml version=\"1.0\"?>" << "\n";
    // header info
    vtkfile << "<!--" << "\n";
    vtkfile << "VTU file produced by JAMS++ (" << QUOTEME(GITCOMMIT) << ")\n";
    vtkfile << "  configuration file: " << seedname << "\n";
    vtkfile << "  iteration: " << solver->iteration() << "\n";
    vtkfile << "  time: " << solver->time() << "\n";
    vtkfile << "  temperature: " << solver->physics()->temperature() << "\n";
    vtkfile << "  applied field: (" << solver->physics()->applied_field(0) << ", " << solver->physics()->applied_field(1) << ", " << solver->physics()->applied_field(2) << ")\n";
    vtkfile << "-->" << "\n";
    vtkfile << "<VTKFile type=\"UnstructuredGrid\" version=\"1.0\"  header_type=\"UInt32\" byte_order=\"LittleEndian\">" << "\n";
    vtkfile << "  <UnstructuredGrid>" << "\n";
    vtkfile << "    <Piece NumberOfPoints=\""<< num_points <<"\"  NumberOfCells=\"1\">" << "\n";
    vtkfile << "      <PointData Scalars=\"scalars\">" << "\n";
    vtkfile << "        <DataArray type=\"Int32\" Name=\"type\" NumberOfComponents=\"1\" format=\"appended\" offset=\"" << 0 << "\" />" << "\n";
    vtkfile << "        <DataArray type=\"Float64\" Name=\"spin\" NumberOfComponents=\"3\" format=\"appended\" RangeMin=\"-1.0\" RangeMax=\"1.0\" offset=\"" << header_bytesize + types_bytesize << "\" />" << "\n";
    vtkfile << "      </PointData>" << "\n";
    vtkfile << "      <CellData>" << "\n";
    vtkfile << "      </CellData>" << "\n";
    vtkfile << "      <Points>" << "\n";
    vtkfile << "        <DataArray type=\"Float32\" NumberOfComponents=\"3\" format=\"appended\" offset=\"" << 2*header_bytesize + spins_bytesize + types_bytesize << "\" />" << "\n";
    vtkfile << "      </Points>" << "\n";
    vtkfile << "      <Cells>" << "\n";
    vtkfile << "        <DataArray type=\"Int32\" Name=\"connectivity\" format=\"ascii\">" << "\n";
    vtkfile << "          1" << "\n";
    vtkfile << "        </DataArray>" << "\n";
    vtkfile << "        <DataArray type=\"Int32\" Name=\"offsets\" format=\"ascii\">" << "\n";
    vtkfile << "          1" << "\n";
    vtkfile << "        </DataArray>" << "\n";
    vtkfile << "        <DataArray type=\"UInt8\" Name=\"types\" format=\"ascii\">" << "\n";
    vtkfile << "          1" << "\n";
    vtkfile << "        </DataArray>" << "\n";
    vtkfile << "      </Cells>" << "\n";
    vtkfile << "    </Piece>" << "\n";
    vtkfile << "  </UnstructuredGrid>" << "\n";
    vtkfile << "<AppendedData encoding=\"raw\">" << "\n";
    vtkfile << "_";

    vtkfile.write(reinterpret_cast<char*>(&types_bytesize), header_bytesize);
    vtkfile.write(reinterpret_cast<char*>(types_binary_data.data()), types_bytesize);
    vtkfile.write(reinterpret_cast<char*>(&spins_bytesize), header_bytesize);
    if (num_slice_points == 0) {
        vtkfile.write(reinterpret_cast<char*>(s.data()), spins_bytesize);
    } else {
        vtkfile.write(reinterpret_cast<char*>(spins_binary_data.data()), spins_bytesize);
    }
    vtkfile.write(reinterpret_cast<char*>(&points_bytesize), header_bytesize);
    vtkfile.write(reinterpret_cast<char*>(points_binary_data.data()), points_bytesize);
    vtkfile << "\n</AppendedData>" << "\n";
    vtkfile << "</VTKFile>" << "\n";
    vtkfile.close();
  }
}

VtuMonitor::~VtuMonitor() {
}
