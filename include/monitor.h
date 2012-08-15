#ifndef __MONITOR_H__
#define __MONITOR_H__


enum ConvergenceType {convNone, convMag, convPhi, convSinPhi};

class Monitor {
  public:
    Monitor() : initialised(false) {}

    virtual ~Monitor(){}

    virtual void initialise();
    virtual void run();
    virtual void write(const double &time);
	
	void initConvergence(ConvergenceType type, const double tol);
	bool checkConvergence();
	

    static Monitor* Create();
  protected:
    bool initialised;

};

#endif // __MONITOR_H__
