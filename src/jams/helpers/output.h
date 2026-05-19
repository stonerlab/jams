//
// Created by Joe Barker on 2018/08/28.
//

#ifndef JAMS_HELPERS_OUTPUT_H
#define JAMS_HELPERS_OUTPUT_H
#include <array>
#include <string>
#include <vector>
#include <sstream>
#include <iomanip>
#include <fstream>
#include <cassert>
#include <cmath>
#include <optional>
#include <stdexcept>
#include <unordered_set>
#include <utility>

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
    void redirect_standard_streams(const std::string& filename);

    /// Open an output file with 'filename' in the full output path if not
    /// already open.
    void open_output_file_just_in_time(std::ofstream& os, const std::string& filename);

    std::string full_path_filename(const std::string& ending);
    std::string full_path_filename_series(const std::string& ending, int num, int width=7);
    std::string monitor_filename(const std::string& monitor_name, const std::string& extension);
    std::string monitor_filename_series(const std::string& monitor_name, const std::string& extension, int num, int width=7);
    std::string hamiltonian_filename(
        const std::string& hamiltonian_name,
        const std::string& output_name,
        const std::string& extension);
    std::string output_path();

    int lock_file(const std::string& lock_filename);
    void unlock_file(int fd);

    enum class ColFmt {
      Scientific,
      Fixed,
      Integer
    };

    struct ColDef {
        std::string name;
        std::string units;
        ColFmt format = ColFmt::Scientific;
    };

    inline void apply_format(std::ostream& os, ColFmt fmt, int precision = 8)
    {
      os.unsetf(std::ios::floatfield);  // Clear scientific/fixed flags first

      switch (fmt) {
        case ColFmt::Scientific:
          os << std::scientific << std::setprecision(precision) << std::setw(precision + 9);
          break;
        case ColFmt::Fixed:
          os << std::fixed << std::setprecision(precision) << std::setw(precision + 4);
          break;
        case ColFmt::Integer:
          os << std::setw(8);
          break;
      }

      os << std::right;
    }

    inline void write_tsv_separator(std::ostream& os, std::size_t index, std::size_t size) {
      if (index + 1 < size) {
        os << '\t';
      }
    }

    inline void validate_tsv_columns(const std::vector<ColDef>& cols) {
      std::unordered_set<std::string> names;

      for (const auto& col : cols) {
        if (col.name.empty()) {
          throw std::runtime_error("TSV column name must not be empty");
        }
        if (!names.insert(col.name).second) {
          throw std::runtime_error("duplicate TSV column name '" + col.name + "'");
        }
      }
    }

    inline void validate_tsv_row_size(std::size_t num_cols, std::size_t num_values) {
      if (num_cols != num_values) {
        throw std::runtime_error(
            "TSV row size mismatch: expected " + std::to_string(num_cols) +
            " values, got " + std::to_string(num_values));
      }
    }

    inline void write_tsv_row(std::ostream& os,
                          const std::vector<ColDef>& cols,
                          const std::vector<double>& values,
                          int precision)
    {
      validate_tsv_row_size(cols.size(), values.size());

      for (std::size_t i = 0; i < cols.size(); ++i) {
        const auto& col = cols[i];
        double v = values[i];

        apply_format(os, col.format, precision);

        switch (col.format) {
          case ColFmt::Integer:
            os << std::llround(v);
            break;
          case ColFmt::Scientific:
          case ColFmt::Fixed:
            os << v;
            break;
        }
        write_tsv_separator(os, i, cols.size());
      }

      os << '\n';
    }

    inline void write_tsv_row(std::ostream& os,
                          const std::vector<ColDef>& cols,
                          const std::vector<std::optional<double>>& values,
                          int precision)
    {
      validate_tsv_row_size(cols.size(), values.size());

      for (std::size_t i = 0; i < cols.size(); ++i) {
        const auto& col = cols[i];

        apply_format(os, col.format, precision);
        if (!values[i].has_value()) {
          os << "--------";
          write_tsv_separator(os, i, cols.size());
          continue;
        }

        const double v = *values[i];
        switch (col.format) {
          case ColFmt::Integer:
            os << std::llround(v);
            break;
          case ColFmt::Scientific:
          case ColFmt::Fixed:
            os << v;
            break;
        }
        write_tsv_separator(os, i, cols.size());
      }

      os << '\n';
    }


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

      for (std::size_t i = 0; i < cols.size(); ++i) {
        const auto& col = cols[i];
        apply_format(os, col.format, precision);
        os << col.name;
        write_tsv_separator(os, i, cols.size());
      }

      os << '\n';
      return os.str();
    }

class TsvWriter {
      public:
        TsvWriter() = default;

        TsvWriter(const std::string& filename, std::vector<ColDef> cols, int precision = 8)
          : file_(filename), cols_(std::move(cols)), precision_(precision) {
          if (!file_) {
            throw std::runtime_error("Failed to open TSV file: " + filename);
          }
          validate_tsv_columns(cols_);
          write_header();
        }

        void open(const std::string& filename, std::vector<ColDef> cols, int precision = 8) {
          file_.close();
          file_.clear();
          file_.open(filename);
          if (!file_) {
            throw std::runtime_error("Failed to open TSV file: " + filename);
          }
          cols_ = std::move(cols);
          precision_ = precision;
          validate_tsv_columns(cols_);
          write_header();
        }

        std::size_t num_cols() const noexcept {
          return cols_.size();
        }

        const std::vector<ColDef>& columns() const  noexcept { return cols_; }
        int precision() const noexcept { return precision_; }

        void set_precision(int precision) noexcept { precision_ = precision; }

        void write_row(const std::vector<double>& values) {
          validate_tsv_row_size(cols_.size(), values.size());
          write_tsv_row(file_, cols_, values, precision_);
        }

        void write_row(const std::vector<std::optional<double>>& values) {
          validate_tsv_row_size(cols_.size(), values.size());
          write_tsv_row(file_, cols_, values, precision_);
        }

        template<typename Container>
        void write_row_container(const Container& c) {
          std::vector<double> values;
          values.reserve(cols_.size());
          for (const auto& v : c) {
            values.push_back(v);
          }
          write_row(values);
        }

        template<typename... Ts>
        void write_row_values(Ts... xs) {
          static_assert(sizeof...(Ts) > 0, "write_row_values needs at least one value");
          std::array<double, sizeof...(Ts)> arr {static_cast<double>(xs)...};
          write_row_container(arr);
        }

        std::ofstream& stream() noexcept { return file_; }

      private:
        void write_header() {
          file_ << make_json_units_string(cols_);
          file_ << make_tsv_header_row(cols_, precision_);
        }

        std::ofstream file_;
        std::vector<ColDef> cols_;
        int precision_ = 8;

    };
}

#endif //JAMS_HELPERS_OUTPUT_H
