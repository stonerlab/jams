#ifndef __MFPT_H__
#define __MFPT_H__

#include <fstream>
#include <libconfig.h++>

#include "physics.h"

class MFPTPhysics : public Physics {
  public:
    MFPTPhysics() 
      : initialised(false),
        maskArray(),
        MFPTFile()
    {}
    ~MFPTPhysics();
    void init(libconfig::Setting &phys);
    void run(double realtime, const double dt);
    virtual void monitor(double realtime, const double dt);
  private:
    bool initialised;
    std::vector<double> maskArray;
    std::ofstream MFPTFile;
};

#endif // __MFPT_H__
