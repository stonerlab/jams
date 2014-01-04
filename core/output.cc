#include "core/output.h"

#include <cstdarg>
#include <cstdio>
#include <string>

void Output::open(const char *fname, ...)
{
  va_list args;

  if(fname == reinterpret_cast<const char*>(NULL)) {
    return;
  }

  va_start(args,fname);
    vsprintf(buffer, fname, args);
  va_end(args);


  logfile.open(buffer);

  if(!logfile.is_open()) {
    fprintf(stderr, "Failed to open log file %s\n", buffer);
  }
}

void Output::close()
{
  logfile.close();
}

void Output::write(const char* string, ...)
{
  va_list args;

  if(string == reinterpret_cast<const char*>(NULL)) {
    return;
  }

  va_start(args,string);
    vsprintf(buffer, string, args);
  va_end(args);

  logfile << buffer;

  if (console == true) {
      print(buffer);
      fflush(stdout);
  }
}

void Output::print(const char* string, ...)
{
  va_list args;

  if(string == reinterpret_cast<const char*>(NULL)) {
    return;
  }

  va_start(args, string);
    vprintf(string, args);
  va_end(args);

  fflush(stdout);
}
