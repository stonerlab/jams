// neighbour_list_interaction.h                                                          -*-C++-*-
#ifndef INCLUDED_JAMS_GENERAL_SPARSE_TWO_SITE_INTERACTION
#define INCLUDED_JAMS_GENERAL_SPARSE_TWO_SITE_INTERACTION

#if HAS_CUDA
#include <jams/cuda/cuda_stream.h>
#endif

#include <jams/core/hamiltonian.h>
#include <jams/containers/sparse_matrix.h>
#include <jams/containers/sparse_matrix_builder.h>
#include <jams/helpers/output.h>
#include <jams/containers/interaction_list.h>

#include <fstream>

template <class T>
class GeneralSparseTwoSiteInteractionHamiltonian : public Hamiltonian {
public:
    GeneralSparseTwoSiteInteractionHamiltonian(const libconfig::Setting &settings, unsigned int size);

    virtual double calculate_total_energy(double time) = 0;

    virtual void calculate_energies(double time) = 0;

    virtual void calculate_fields(double time) = 0;

    virtual Vec3 calculate_field(int i, double time) = 0;

    virtual double calculate_energy(int i, double time) = 0;

    virtual double calculate_energy_difference(int i, const Vec3 &spin_initial, const Vec3 &spin_final, double time) = 0;

protected:
    void insert_interaction(int i, int j, const T &value);

    // finishes constructing the sparse_matrix_builder_ making the builder
    // emit a matrix for use in calculations
    void finalize(jams::SparseMatrixSymmetryCheck symmetry_check);

    const typename jams::SparseMatrix<T>& interaction_matrix() const;

private:
    bool is_finalized_ = false; // is the sparse matrix finalized and built
    typename jams::SparseMatrix<T>::Builder sparse_matrix_builder_; // helper to build the sparse matrix and output a chosen type
    typename jams::SparseMatrix<T> interaction_matrix_; // the sparse matrix to be used in calculations

    #if HAS_CUDA
    CudaStream cusparse_stream_; // cuda stream to run in
    #endif
};


template<class T>
const jams::SparseMatrix<T> &
GeneralSparseTwoSiteInteractionHamiltonian<T>::interaction_matrix() const {
  return interaction_matrix_;
}


template <class T>
GeneralSparseTwoSiteInteractionHamiltonian<T>::GeneralSparseTwoSiteInteractionHamiltonian(const libconfig::Setting &settings, const unsigned int size)
    : Hamiltonian(settings, size),
    sparse_matrix_builder_(size, size)
{}

template <class T>
void GeneralSparseTwoSiteInteractionHamiltonian<T>::insert_interaction(const int i, const int j, const T &value) {
  assert(!is_finalized_);

  sparse_matrix_builder_.insert(i, j, value);
}

template <class T>
void GeneralSparseTwoSiteInteractionHamiltonian<T>::finalize(jams::SparseMatrixSymmetryCheck symmetry_check) {
  assert(!is_finalized_);

  if (debug_is_enabled()) {
    std::ofstream os(jams::output::full_path_filename("DEBUG_" + name() + "_spm.tsv"));
    sparse_matrix_builder_.output(os);
    os.close();
  }

  switch(symmetry_check) {
    case jams::SparseMatrixSymmetryCheck::None:
      break;
    case jams::SparseMatrixSymmetryCheck::Symmetric:
      if (!sparse_matrix_builder_.is_symmetric()) {
        throw std::runtime_error("sparse matrix for " + name() + " is not symmetric");
      }
      break;
    case jams::SparseMatrixSymmetryCheck::StructurallySymmetric:
      if (!sparse_matrix_builder_.is_structurally_symmetric()) {
        throw std::runtime_error("sparse matrix for " + name() + " is not structurally symmetric");
      }
      break;
  }

  interaction_matrix_ = sparse_matrix_builder_
      .set_format(jams::SparseMatrixFormat::CSR)
      .build();
  std::cout << "  " << name() << " sparse matrix memory (CSR): " << memory_in_natural_units(interaction_matrix_.memory()) << "\n";
  sparse_matrix_builder_.clear();
  is_finalized_ = true;
}

#endif

// ----------------------------- END-OF-FILE ----------------------------------