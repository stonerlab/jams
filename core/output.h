class Output;

#ifndef __OUTPUT_H__
#define __OUTPUT_H__

#include <iostream>
#include <fstream>

class Output {

  public:
    Output() : logfile(), console(true) { enableConsole(); }

    Output(const char *fname) 
      : logfile(), console(true) 
    {
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
    bool console;
    char buffer[1024];

};

#endif // __OUTPUT_H__
