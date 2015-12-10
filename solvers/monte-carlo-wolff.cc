// Copyright 2015 Joseph Barker. All rights reserved.

#include "solvers/monte-carlo-wolff.h"

MonteCarloWolffSolver::MonteCarloWolffSolver() {
}

void MonteCarloWolffSolver::initialize(int argc, char **argv, double idt) {
  // initialize base class
  Solver::initialize(argc, argv, idt);

}

void MonteCarloWolffSolver::run() {
  iteration_++;
}

MonteCarloWolffSolver::~MonteCarloWolffSolver() {

}