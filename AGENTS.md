# AGENTS.md

Guidance for agents working in this repository.

## Project Overview

JAMS is Joe's Awesome Magnetic Simulator, an atomistic magnetisation dynamics code with optional CUDA GPU acceleration. The codebase is primarily C++20 and CUDA C++20, built with CMake.

## Repository Layout

- `src/jams/`: main simulator source code.
- `src/jams/solvers/`: CPU and CUDA solver implementations.
- `src/jams/helpers/`: shared utility and numerical helper code.
- `src/jams/hamiltonian/`: classes for implementing spin Hamiltonian terms.
- `src/jams/monitors/`: classes implementing the monitor design pattern to analyse the simulation.
- `src/jams/test/`: low-level/unit test targets.
- `test/`: Python integration tests and test data.
- `cmake/`: CMake modules and dependency configuration.
- `docs/`: documentation sources.
- `scripts/`: project build, docs, and integration-test helpers.

## Build and Test Commands

Use out-of-source builds only. The top-level CMake configuration rejects in-source builds.

```sh
cmake -S . -B cmake-build-debug -DCMAKE_BUILD_TYPE=Debug -DJAMS_BUILD_CUDA=OFF -DJAMS_BUILD_TESTS=ON
cmake --build cmake-build-debug --parallel
```

For integration tests, prefer the project script:

```sh
bash scripts/run-integration-tests.sh
```

Useful integration-test variants:

```sh
bash scripts/run-integration-tests.sh -o -DJAMS_BUILD_CUDA=OFF -o -DJAMS_BUILD_OMP=OFF
bash scripts/run-integration-tests.sh --build-type Debug --generator Ninja --jobs 8
bash scripts/run-integration-tests.sh --tests test/test_exchange_symops.py
```

When validating command-line configuration parsing manually:

```sh
./cmake-build-debug/bin/jams input.cfg --validate-config
```

When running unit tests, by default remove any test marked "SLOW"

If you cannot find a CUDA device then ask for authorisation to check for one outside of the sandbox.

## Coding Notes

- Follow the existing C++/CUDA style in nearby files before introducing new patterns.
- Keep numerical changes conservative and document unit assumptions when they are not obvious.
- Internal units used by the code include picoseconds for time, terahertz for frequency, millielectron volts for energy, and Tesla for field.
- Avoid broad refactors when fixing focused behavior.
- Prefer existing helpers in `src/jams/helpers/` over adding duplicate utility code.
- For CUDA work, keep CPU-only builds in mind and guard CUDA-specific code behind the existing build structure.
- Even if CUDA is not enabled in a build, check whether CUDA code needs refactoring when CPU code is altered.
- When reviewing code, always consider the physics and mathematics as well as the code structure.

## Dependency and Generated Files

- CMake may fetch or build dependencies when `JAMS_BUILD_IN_DEPENDENCIES=ON`.
- Do not commit local build directories such as `cmake-build-*`, virtual environments, or generated test output unless explicitly requested.

## Git Hygiene

- Check `git status` before editing and preserve unrelated user changes.
- Keep commits scoped to the requested change.
- Do not rewrite history, reset, or discard changes unless explicitly asked.
