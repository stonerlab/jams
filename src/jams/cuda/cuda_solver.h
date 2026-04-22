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
   }
  bool is_cuda_solver() const override { return true; }
  void compute_fields() override;

    void notify_monitors() override
    {
        std::vector<Monitor*> host_due_monitors;
        std::vector<Monitor*> device_due_monitors;
        host_due_monitors.reserve(monitors_.size());
        device_due_monitors.reserve(monitors_.size());

        for (auto& m : monitors_) {
            if (m->is_updating(iteration_)) {
                if (m->access_mode() == Monitor::AccessMode::kDeviceOnly) {
                    device_due_monitors.push_back(m.get());
                } else {
                    host_due_monitors.push_back(m.get());
                }
            }
        }

        if (host_due_monitors.empty() && device_due_monitors.empty()) {
            return;
        }

        for (auto* monitor : device_due_monitors) {
            monitor->update(*this);
        }

        if (host_due_monitors.empty()) {
            return;
        }

        synchronize_on_spin_barrier_event();
        for (auto* monitor : host_due_monitors) {
            monitor->update(*this);
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
        cudaEventCreate(&spin_barrier_event_, cudaEventDisableTiming);
        DEBUG_CHECK_CUDA_ASYNC_STATUS
    }

    jams::Real** dev_field_ptrs_ = nullptr;
    cudaEvent_t spin_barrier_event_ {};
};

#endif  // JAMS_CORE_CUDASOLVER_H
