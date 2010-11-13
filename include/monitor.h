#ifndef __MONITOR_H__
#define __MONITOR_H__

class Monitor {
  public:
    Monitor() : initialised(false) {}

    virtual ~Monitor(){}

    virtual void initialise();
    virtual void run();
    virtual void write(const double &time);

    static Monitor* Create();
  protected:
    bool initialised;

};

#endif // __MONITOR_H__
