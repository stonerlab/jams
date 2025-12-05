//
// Created by Joe Barker on 2018/08/28.
//

#ifndef JAMS_HELPERS_OUTPUT_H
#define JAMS_HELPERS_OUTPUT_H
#include <string>
#include <vector>
#include <sstream>
#include <iomanip>

namespace jams::fmt {
struct SciFmt {
  int precision;
};

inline SciFmt sci(int p) { return SciFmt{p}; }

inline std::ostream& operator<<(std::ostream& os, const SciFmt& s) {
  // Width is 8 wider than precision to allow
  // <space><sign><number><.><precision...><e><sign><0-9><0-9><0-9>
  os << std::setprecision(s.precision)
     << std::setw(s.precision + 9)
     << std::scientific
     << std::right;
  return os;
}

inline std::ostream &integer(std::ostream &os) {
  return os << std::fixed << std::setw(8) << std::right;
}

inline std::ostream &fixed_integer(std::ostream &os) {
  return os << std::fixed << std::setw(8);
}

inline std::ostream &decimal(std::ostream &os) {
  return os << std::setprecision(6) << std::setw(16) << std::fixed << std::right;
}

inline std::ostream &sci(std::ostream &os) {
  return os << std::setprecision(8) << std::setw(16) << std::scientific << std::right;
};

inline std::ostream &header(std::ostream &os) {
  return os << std::setw(16) << std::right;
};
}

namespace jams::output {
    void initialise();

    /// Open an output file with 'filename' in the full output path if not
    /// already open.
    void open_output_file_just_in_time(std::ofstream& os, const std::string& filename);

    std::string full_path_filename(const std::string& ending);
    std::string full_path_filename_series(const std::string& ending, int num, int width=7);
    std::string output_path();

    int lock_file(const std::string& lock_filename);
    void unlock_file(int fd);

    struct ColDef {
        std::string name;
        std::string units;
    };

    inline std::string make_json_units_string(const std::vector<ColDef>& cols) {
      std::ostringstream os;
      os << "# { ";

      for (std::size_t i = 0; i < cols.size(); ++i) {
        if (i > 0) {
          os << ", ";
        }
        os << '"' << cols[i].name << "\": \"" << cols[i].units << '"';
      }

      os << " }\n";
      return os.str();
    }

    inline std::string make_tsv_header_row(const std::vector<ColDef>& cols, int precision) {
      std::ostringstream os;
      os << std::scientific << std::setprecision(precision) << std::right;

      for (const auto& col : cols) {
        os << fmt::sci(precision) << col.name;
      }

      os << '\n';
      return os.str();
    }
}

#endif //JAMS_HELPERS_OUTPUT_H
