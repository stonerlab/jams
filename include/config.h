class Config;

#ifndef __CONFIG_H__
#define __CONFIG_H__

#include <libconfig.h++>


class Config
{
  public:
    void open(const char *fname, ...);

  private:
    libconfig::Config cfg;
    char buffer[1024];
};

#endif // __CONFIG_H__
