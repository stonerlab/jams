#ifndef __NOISE_H__
#define __NOISE_H__

enum NoiseType {WHITE, OU, FFT};

class Noise
{
  public:
    Noise()
      : initialised(false), temperature(0) {}

    virtual ~Noise(){}

    virtual void initialise(double dt);
    virtual void run();

    static Noise* Create();
    static Noise* Create(NoiseType type);

  protected:
    bool initialised;
    double temperature;
};

#endif // __SOLVER_H__
