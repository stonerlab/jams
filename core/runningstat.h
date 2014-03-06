// Copyright 2014 Joseph Barker. All rights reserved.

#ifndef JAMS_CORE_RUNNINGSTAT_H
#define JAMS_CORE_RUNNINGSTAT_H

#include <cmath>

// Based on class from http://www.johndcook.com/standard_deviation.html
class RunningStat{
 public:
  RunningStat() : m_n(0) {}

  void clear() { m_n = 0; }

  void push(double x) {
    m_n++;
    if (m_n == 1) {
      m_oldM = m_newM = x;
      m_oldS = 0.0;
    } else {
      m_newM = m_oldM + (x - m_oldM)/static_cast<double>(m_n);
      m_newS = m_oldS + (x - m_oldM)*(x - m_newM);

      m_oldM = m_newM;
      m_oldS = m_newS;
    }
  }

  int numDataValues() const { return m_n; }

  double mean() const { return (m_n > 0) ? m_newM : 0.0; }

  double var() const { return ( (m_n > 1) ? m_newS/(m_n - 1) : 0.0 ); }

  double stdDev() const { return sqrt( var() ); }

  double stdErr() const { return sqrt( var() / static_cast<double>(m_n) ); }

 private:
  unsigned long m_n;
  double m_oldM, m_newM, m_oldS, m_newS;
};

#endif  // JAMS_CORE_RUNNINGSTAT_H
