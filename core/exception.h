#ifndef JAMS_CORE_EXCEPTION_H
#define JAMS_CORE_EXCEPTION_H

#include <stdexcept>

class general_exception : std::runtime_error {
    std::string msg;
public:
    general_exception(const std::string &arg, const char *file, int line, const char *function="") :
    std::runtime_error(arg) {
        std::ostringstream o;
        if (strcmp(function, "") == 0) {
            o << file << ":" << line << ": " << arg ;
        } else {
            o << file << ":" << line << "[" << function << "]" << ": " << arg ;
        }
        msg = o.str();
    }
    ~general_exception() throw() {}
    const char *what() const throw() {
        return msg.c_str();
    }
};

#endif  // JAMS_CORE_EXCEPTION_H
