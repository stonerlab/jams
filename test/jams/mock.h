//
// Created by Joseph Barker on 2019-01-29.
//

#ifndef JAMS_MOCK_H
#define JAMS_MOCK_H

#include "jams/core/globals.h"
#include "jams/core/lattice.h"
#include "jams/core/solver.h"
#include "jams/core/physics.h"

class MockJamsTest : public ::testing::Test {
protected:
    // You can remove any or all of the following functions if its body
    // is empty.

    MockJamsTest() {
      // You can do set-up work for each test here.
      ::lattice = new Lattice();
      ::config = new libconfig::Config();
    }

    ~MockJamsTest() = default;

    // If the constructor and destructor are not enough for setting up
    // and cleaning up each test, you can define the following methods:

    void SetUp(const std::string &config_string) {
      // Code here will be called immediately after the constructor (right
      // before each test).
      ::config->readString(config_string);
      ::lattice->init_from_config(*::config);
      ::solver = Solver::create(config->lookup("solver"));
      ::solver->initialize(config->lookup("solver"));
      ::solver->register_physics_module(Physics::create(config->lookup("physics")));
    }

    virtual void TearDown() {
      // Code here will be called immediately after each test (right
      // before the destructor).
      delete ::solver;
      delete ::lattice;
      delete ::config;
    }

    // Objects declared here can be used by all tests in the test case for Foo.
};


#endif //JAMS_MOCK_H
