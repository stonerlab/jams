// Copyright 2014 Joseph Barker. All rights reserved.

#ifndef JAMS_CORE_CUDASOLVER_H
#define JAMS_CORE_CUDASOLVER_H

#include "jams/common.h"
#include "jams/core/solver.h"

class CudaSolver : public Solver {
 public:

   ~CudaSolver() override {
       if (dev_field_ptrs_)
       {
           cudaFree(dev_field_ptrs_);
       }
       if (spin_barrier_event_)
       {
           cudaEventDestroy(spin_barrier_event_);
       }
#if DO_MIXED_PRECISION
       if (field_spin_cache_event_)
       {
           cudaEventDestroy(field_spin_cache_event_);
       }
#endif
   }
  bool is_cuda_solver() const override { return true; }
  void compute_fields() override;
  const jams::MultiArray<jams::Real, 2>& spin_array_for_fields() override;

    void notify_monitors() override
    {
        synchronize_on_spin_barrier_event();
        for (auto& m : monitors_) {
            if (m->is_updating(iteration_)) {
                m->update(*this);
            }
        }
    }

    void record_spin_barrier_event()
    {
        if (!spin_barrier_event_) create_spin_barrier_event();
        assert(spin_barrier_event_);
        cudaEventRecord(spin_barrier_event_, jams::instance().cuda_master_stream().get());
        DEBUG_CHECK_CUDA_ASYNC_STATUS
    }

    void wait_on_spin_barrier_event(cudaStream_t external)
    {
        if (!spin_barrier_event_) create_spin_barrier_event();
        assert(spin_barrier_event_);
        cudaStreamWaitEvent(external, spin_barrier_event_, 0);
        DEBUG_CHECK_CUDA_ASYNC_STATUS
    }

    void synchronize_on_spin_barrier_event()
    {
        if (!spin_barrier_event_) create_spin_barrier_event();
        assert(spin_barrier_event_);
        cudaEventSynchronize(spin_barrier_event_);
        DEBUG_CHECK_CUDA_ASYNC_STATUS
    }
  private:
    void create_spin_barrier_event()
    {
        cudaEventCreateWithFlags(&spin_barrier_event_, cudaEventDisableTiming);
        DEBUG_CHECK_CUDA_ASYNC_STATUS
    }

#if DO_MIXED_PRECISION
    const jams::MultiArray<jams::Real, 2>& refresh_field_spin_array_async();

    void create_field_spin_cache_event()
    {
        cudaEventCreateWithFlags(&field_spin_cache_event_, cudaEventDisableTiming);
        DEBUG_CHECK_CUDA_ASYNC_STATUS
    }

    void wait_on_field_spin_cache_event(cudaStream_t external)
    {
        if (!field_spin_cache_event_) create_field_spin_cache_event();
        assert(field_spin_cache_event_);
        cudaStreamWaitEvent(external, field_spin_cache_event_, 0);
        DEBUG_CHECK_CUDA_ASYNC_STATUS
    }
#endif

    jams::Real** dev_field_ptrs_ = nullptr;
    cudaEvent_t spin_barrier_event_ {};
#if DO_MIXED_PRECISION
    cudaEvent_t field_spin_cache_event_ {};
#endif
};

#endif  // JAMS_CORE_CUDASOLVER_H
