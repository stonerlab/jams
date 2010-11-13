#ifndef __MAGNETISATION_H__
#define __MAGNETISATION_H__

#include "monitor.h"
#include "array2d.h"
#include <fstream>


class MagnetisationMonitor : public Monitor {
  public:
    MagnetisationMonitor()
      : mag(0,0),
        outfile()
    {}

    ~MagnetisationMonitor();

    void initialise();
    void run();
    void write(const double &time);
  private:
    Array2D<double> mag;
    std::ofstream outfile;
};

#endif // __MAGNETISATION_H__

