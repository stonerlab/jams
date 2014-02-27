// Copyright 2014 Joseph Barker. All rights reserved.

class Output;

#ifndef JAMS_CORE_OUTPUT_H
#define JAMS_CORE_OUTPUT_H

#include <fstream>
#include <iostream>
#include <string>

class Output {
 public:
  Output() : logfile(), console(true) { enableConsole(); }

  explicit Output(const char *fname) : logfile(), console(true) {
    open(fname);
    enableConsole();
  }

  ~Output() {
    close();
  }


  void open(const char *fname, ...);
  void close();

  void write(const char* string, ...);
  void print(const char* string, ...);

  void enableConsole() { console = true; }
  void disableConsole() { console = false; }

 private:
  std::ofstream logfile;
  bool console;
  char buffer[1024];
};

#endif  // JAMS_CORE_OUTPUT_H