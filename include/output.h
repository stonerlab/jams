class Output;

#ifndef __OUTPUT_H__
#define __OUTPUT_H__

#include <iostream>
#include <fstream>

class Output {

  public:
    Output() { enableConsole(); }

    Output(const char *fname) {
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

    void enableConsole(){ console = true; }
    void disableConsole(){ console = false; }

  private:
    std::ofstream logfile;
    char buffer[1024];

    bool console;

};

#endif // __OUTPUT_H__
