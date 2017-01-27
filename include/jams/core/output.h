// Copyright 2014 Joseph Barker. All rights reserved.

class Output;

#ifndef JAMS_CORE_OUTPUT_H
#define JAMS_CORE_OUTPUT_H

#include <fstream>
#include <iostream>
#include <string>

enum OutputFormat {TEXT, HDF5};

class Output {
 public:
  Output() : logfile(), console(true), verbose_(false) { enableConsole(); }

  explicit Output(const char *fname) : logfile(), console(true), verbose_(false) {
    open(fname);
    enableConsole();
  }

  ~Output() {
    close();
  }


  void open(const char *fname, ...);
  void close();

  void write(const char* string, ...);
  void verbose(const char* string, ...);
  void print(const char* string, ...);

  bool is_verbose() { return verbose_; };

  void enableConsole() { console = true; }
  void disableConsole() { console = false; }

  void enableVerbose() { verbose_ = true; }
  void disableVerbose() { verbose_ = false; }

 private:
  std::ofstream logfile;
  bool console;
  bool verbose_;
  char buffer[1024];
};

#endif  // JAMS_CORE_OUTPUT_H
