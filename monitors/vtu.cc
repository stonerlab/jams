// Copyright 2014 Joseph Barker. All rights reserved.

#include <cmath>
#include <string>

#include "core/globals.h"
#include "core/lattice.h"
#include "core/utils.h"

#include "monitors/vtu.h"

#define QUOTEME_(x) #x
#define QUOTEME(x) QUOTEME_(x)

VtuMonitor::VtuMonitor(const libconfig::Setting &settings)
: Monitor(settings) {
  using namespace globals;
  ::output.write("\nInitialising Vtu monitor...\n");

  points_binary_data.resize(num_spins, 3);

  for (int i = 0; i < num_spins; ++i) {
    for (int j = 0; j < 3; ++j) {
      points_binary_data(i, j) = lattice.lattice_parameter_*lattice.lattice_positions_[i][j];
    }
  }

  is_equilibration_monitor_ = false;
  output_step_freq_ = settings["output_steps"];
}

void VtuMonitor::update(const int &iteration, const double &time, const double &temperature, const jblib::Vec3<double> &applied_field) {
  using namespace globals;

  if (iteration%output_step_freq_ == 0) {
    int outcount = iteration/output_step_freq_;  // int divisible by modulo above

    const int num_materials = lattice.num_materials();

    std::ofstream vtkfile(std::string(seedname+"_"+zero_pad_number(outcount)+".vtu").c_str());

    uint32_t header_bytesize = sizeof(uint32_t);
    uint32_t type_bytesize   = num_spins*sizeof(int32_t);
    uint32_t points_bytesize = num_spins3*sizeof(float);
    uint32_t spins_bytesize  = num_spins3*sizeof(double);

    vtkfile << "<?xml version=\"1.0\"?>" << "\n";
    // header info
    vtkfile << "<!--" << "\n";
    vtkfile << "VTU file produced by JAMS++ (" << QUOTEME(GITCOMMIT) << ")\n";
    vtkfile << "  configuration file: " << seedname << "\n";
    vtkfile << "  iteration: " << iteration << "\n";
    vtkfile << "  time: " << time << "\n";
    vtkfile << "  temperature: " << temperature << "\n";
    vtkfile << "  applied field: (" << applied_field.x << ", " << applied_field.y << ", " << applied_field.z << ")\n";
    vtkfile << "-->" << "\n";
    vtkfile << "<VTKFile type=\"UnstructuredGrid\" version=\"1.0\"  header_type=\"UInt32\" byte_order=\"LittleEndian\">" << "\n";
    vtkfile << "  <UnstructuredGrid>" << "\n";
    vtkfile << "    <Piece NumberOfPoints=\""<< num_spins <<"\"  NumberOfCells=\"1\">" << "\n";
    vtkfile << "      <PointData Scalars=\"scalars\">" << "\n";
    vtkfile << "        <DataArray type=\"Int32\" Name=\"type\" NumberOfComponents=\"1\" format=\"appended\" offset=\"" << 0 << "\" />" << "\n";
    vtkfile << "        <DataArray type=\"Float64\" Name=\"spin\" NumberOfComponents=\"3\" format=\"appended\" RangeMin=\"-1.0\" RangeMax=\"1.0\" offset=\"" << header_bytesize + type_bytesize << "\" />" << "\n";
    vtkfile << "      </PointData>" << "\n";
    vtkfile << "      <CellData>" << "\n";
    vtkfile << "      </CellData>" << "\n";
    vtkfile << "      <Points>" << "\n";
    vtkfile << "        <DataArray type=\"Float32\" NumberOfComponents=\"3\" format=\"appended\" offset=\"" << 2*header_bytesize + spins_bytesize + type_bytesize << "\" />" << "\n";
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

    vtkfile.write(reinterpret_cast<char*>(&type_bytesize), header_bytesize);
    vtkfile.write(reinterpret_cast<char*>(&lattice.lattice_material_num_[0]), type_bytesize);
    vtkfile.write(reinterpret_cast<char*>(&spins_bytesize), header_bytesize);
    vtkfile.write(reinterpret_cast<char*>(s.data()), spins_bytesize);
    vtkfile.write(reinterpret_cast<char*>(&points_bytesize), header_bytesize);
    vtkfile.write(reinterpret_cast<char*>(points_binary_data.data()), points_bytesize);
    vtkfile << "\n</AppendedData>" << "\n";
    vtkfile << "</VTKFile>" << "\n";
    vtkfile.close();
  }
}

VtuMonitor::~VtuMonitor() {
}
