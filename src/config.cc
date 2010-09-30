#include "config.h"
#include "globals.h"

#include <libconfig.h++>

void Config::open(const char * fname, ...){
  va_list args;

  if(fname == reinterpret_cast<const char*>(NULL)) {
    return;
  }

  va_start(args,fname);
    vsprintf(buffer, fname, args);
  va_end(args);

  try
  {
    cfg.readFile(buffer);
  }
  catch(const libconfig::FileIOException &fioex)
  {
    jams_error("Failed to read config file %s",buffer);
  }
  catch(const libconfig::ParseException &pex)
  {
    jams_error("Config parse error at %s:%i - %s", pex.getFile(),
      pex.getLine(), pex.getError());
  }
}
