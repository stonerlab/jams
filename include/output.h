class Output;

#ifndef __OUTPUT_H__
#define __OUTPUT_H__

#include <iostream>
#include <fstream>

class Output {

  public:
    Output() {}

    Output(const char *fname) {
      open(fname);
    }

    ~Output() {
      close();
    }

    void open(const char *fname, ...);
    void close();

    void write(const char* string, ...);
    void print(const char* string, ...);

  private:
    std::ofstream logfile;
    char buffer[1024];

};

#endif // __OUTPUT_H__
