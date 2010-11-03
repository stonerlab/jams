#include "monitor.h"
#include "globals.h"

void Monitor::initialise() {
  if(initialised == true) {
    jams_error("Monitor is already initialised");
  }

  output.write("Initialising monitor\n");
}

void Monitor::run() {

}

void Monitor::write() {

}
