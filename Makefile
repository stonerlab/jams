# default target
all::

# Define V=1 for verbose output
#V=1
# Define SHELL_PATH if sh is not in /bin/sh
#
# Define LIBCONFIGDIR if the libconfig header and library files are in
# /foo/bar/include and /foo/bar/lib directories.
# LIBCONFIGDIR=/Users/jbarker/local
# Define NO_CUDA if you wish to build a binary without CUDA support
#NO_CUDA=1
#
# Define CUDADIR if the cuda header and library files are in
# /foo/bar/include and /foo/bar/lib directories.
CUDADIR=/usr/local/cuda
# Define MKLROOT if the mkl root path is not set in the environment
#
# Define CUDA_BUILD_FERMI if you want to build support for Fermi architechture
# cards (Compute 2.0/2.1 - GF100, GF110, GF104, GF106, GF108, GF114, GF116, GF119):
#
# GeForce GTX 590, GeForce GTX 580, GeForce GTX 570, GeForce GTX 480,
# GeForce GTX 470, GeForce GTX 465, GeForce GTX 480M, Quadro 6000, Quadro 5000,
# Quadro 4000, Quadro 4000 for Mac, Quadro Plex 7000, Quadro 5010M, Quadro
# 5000M, Tesla C2075, Tesla C2050/C2070, Tesla M2050/M2070/M2075/M2090, GeForce
# GTX 560 Ti, GeForce GTX 550 Ti, GeForce GTX 460, GeForce GTS 450, GeForce GTS
# 450*, GeForce GT 640 (GDDR3), GeForce GT 630, GeForce GT 620, GeForce GT 610,
# GeForce GT 520, GeForce GT 440, GeForce GT 440*, GeForce GT 430, GeForce GT
# 430*, GeForce GTX 675M, GeForce GTX 670M, GeForce GT 635M, GeForce GT 630M,
# GeForce GT 625M, GeForce GT 720M, GeForce GT 620M, GeForce 710M, GeForce 610M,
# GeForce GTX 580M, GeForce GTX 570M, GeForce GTX 560M, GeForce GT 555M, GeForce
# GT 550M, GeForce GT 540M, GeForce GT 525M, GeForce GT 520MX, GeForce GT 520M,
# GeForce GTX 485M, GeForce GTX 470M, GeForce GTX 460M, GeForce GT 445M, GeForce
# GT 435M, GeForce GT 420M, GeForce GT 415M, GeForce 710M, GeForce 410M, Quadro
# 2000, Quadro 2000D, Quadro 600, Quadro 410, Quadro 4000M, Quadro 3000M, Quadro
# 2000M, Quadro 1000M, NVS 5400M, NVS 5200M, NVS 4200M
# CUDA_BUILD_FERMI=1

# Define CUDA_BUILD_KEPLAR if you want to build support for Keplar architechture
# cards (Compute 3.0/3.5 - GK104, GK106, GK107, GK110, GK208)
CUDA_BUILD_KEPLAR=1
# GeForce GTX 770, GeForce GTX 760, GeForce GTX 690, GeForce GTX 680, GeForce
# GTX 670, GeForce GTX 660 Ti, GeForce GTX 660, GeForce GTX 650 Ti BOOST,
# GeForce GTX 650 Ti, GeForce GTX 650, GeForce GTX 780M, GeForce GTX 770M,
# GeForce GTX 765M, GeForce GTX 760M, GeForce GTX 680MX, GeForce GTX 680M,
# GeForce GTX 675MX, GeForce GTX 670MX, GeForce GTX 660M, GeForce GT 750M,
# GeForce GT 650M, GeForce GT 745M, GeForce GT 645M, GeForce GT 740M, GeForce GT
# 730M, GeForce GT 640M, GeForce GT 640M LE, GeForce GT 735M, GeForce GT 730M,
# Quadro K5000, Quadro K4000, Quadro K2000, Quadro K2000D, Quadro K600, Quadro
# K500M, Tesla K10, GeForce GTX TITAN, GeForce GTX TITAN Black, GeForce GTX 780
# Ti, GeForce GTX 780, GeForce GT 640 (GDDR5), GeForce GT 630 v2, Quadro K6000,
# Tesla K40, Tesla K20x, Tesla K20
#
# Define CUDA_BUILD_MAXWELL if you want to build support for Maxwell architechture
# cards (Compute 5.0 - GM107, GM108)
#
# GeForce GTX 750 Ti, GeForce GTX 750 , GeForce GTX 860M, GeForce GTX 850M,
# GeForce 840M, GeForce 830M
#CUDA_BUILD_MAXWELL=1

GITCOMMIT = $(shell git rev-parse HEAD)
GITSHORT = $(shell git rev-parse --short HEAD)
CPUTYPE = $(shell uname -m | sed "s/\\ /_/g")
SYSTYPE = $(shell uname -s)

CFLAGS =  -std=c++11 -march=native -O3 -g -funroll-loops -Wall -DNDEBUG -DGITCOMMIT="$(GITCOMMIT)"
CUFLAGS =
LDFLAGS =
ALL_CUFLAGS = $(CUFLAGS)
ALL_CFLAGS = $(CPPFLAGS) $(CFLAGS)
ALL_LDFLAGS = $(LDFLAGS)

BASIC_CFLAGS = -I. -I/usr/local/include -I/usr/local/opt/openblas/include
BASIC_CUFLAGS = -I. -I$(CUDADIR)/include
BASIC_LDFLAGS = -L/usr/local/lib -L/usr/local/opt/openblas/lib

CC = g++
NVCC = nvcc
RM = rm -f

ifndef V
	QUIET_CC   = @echo '   ' CC $@;
	QUIET_NVCC = @echo '   ' NVCC $@;
	QUIET_LINK = @echo '   ' LINK $@;
	export V
endif

OBJS += core/jams++.o
OBJS += core/lattice.o
OBJS += core/maths.o
OBJS += core/monitor.o
OBJS += core/output.o
OBJS += core/physics.o
OBJS += core/rand.o
OBJS += core/stats.o
OBJS += core/solver.o
OBJS += core/sparsematrix.o
OBJS += core/thermostat.o
OBJS += core/hamiltonian.o
OBJS += monitors/boltzmann.o
OBJS += monitors/energy.o
OBJS += monitors/spin_pumping.o
OBJS += monitors/magnetisation.o
OBJS += monitors/magnetisation_rate.o
OBJS += monitors/structurefactor.o
OBJS += monitors/torque.o
OBJS += monitors/vtu.o
OBJS += monitors/hdf5.o
OBJS += monitors/xyz.o
OBJS += monitors/binary.o
OBJS += monitors/skyrmion.o
OBJS += physics/fieldcool.o
OBJS += physics/fmr.o
OBJS += physics/mfpt.o
OBJS += physics/square.o
OBJS += physics/ttm.o
OBJS += physics/ping.o
OBJS += solvers/heunllg.o
OBJS += solvers/metropolismc.o
OBJS += solvers/constrainedmc.o
OBJS += solvers/monte-carlo-wolff.o
OBJS += hamiltonian/dipole.o
OBJS += hamiltonian/dipole_bruteforce.o
OBJS += hamiltonian/dipole_tensor.o
OBJS += hamiltonian/dipole_ewald.o
OBJS += hamiltonian/dipole_fft.o

HDR += core/consts.h
HDR += core/error.h
HDR += core/geometry.h
HDR += core/globals.h
HDR += core/lattice.h
HDR += core/maths.h
HDR += core/monitor.h
HDR += core/output.h
HDR += core/physics.h
HDR += core/rand.h
HDR += core/stats.h
HDR += core/runningstat.h
HDR += core/solver.h
HDR += core/sparsematrix4d.h
HDR += core/sparsematrix.h
HDR += core/thermostat.h
HDR += core/utils.h
HDR += core/slice.h
HDR += core/hamiltonian.h
HDR += monitors/boltzmann.h
HDR += monitors/energy.h
HDR += monitors/spin_pumping.h
HDR += monitors/magnetisation.h
HDR += monitors/magnetisation_rate.h
HDR += monitors/structurefactor.h
HDR += monitors/torque.h
HDR += monitors/vtu.h
HDR += monitors/hdf5.h
HDR += monitors/xyz.h
HDR += monitors/binary.h
HDR += monitors/skyrmion.h
HDR += physics/empty.h
HDR += physics/fieldcool.h
HDR += physics/fmr.h
HDR += physics/mfpt.h
HDR += physics/square.h
HDR += physics/ttm.h
HDR += physics/ping.h
HDR += solvers/heunllg.h
HDR += solvers/metropolismc.h
HDR += solvers/constrainedmc.h
HDR += solvers/monte-carlo-wolff.h
HDR += hamiltonian/dipole.h
HDR += hamiltonian/dipole_bruteforce.h
HDR += hamiltonian/dipole_tensor.h
HDR += hamiltonian/dipole_ewald.h
HDR += hamiltonian/dipole_fft.h
HDR += hamiltonian/uniaxial.h
HDR += hamiltonian/exchange.h
HDR += hamiltonian/zeeman.h

ifndef NO_CUDA
	CUDA_OBJS += core/cuda_solver.o
	CUDA_OBJS += core/cuda_sparsematrix.o
	CUDA_OBJS += core/cuda_solver_kernels.o
	CUDA_OBJS += core/cuda_array_kernels.o
	CUDA_OBJS += solvers/cuda_heunllg.o
	CUDA_OBJS += solvers/cuda_constrainedmc.o
	CUDA_OBJS += thermostats/cuda_langevin_coth.o
	OBJS += thermostats/cuda_langevin_white.o
	CUDA_OBJS += hamiltonian/uniaxial.o
	CUDA_OBJS += hamiltonian/zeeman.o
	CUDA_OBJS += hamiltonian/exchange.o

	CUDA_HDR += core/cuda_defs.h
	CUDA_HDR += core/cuda_solver.h
	CUDA_HDR += core/cuda_sparsematrix.h
	CUDA_HDR += core/cuda_solver_kernels.h
	CUDA_HDR += core/cuda_array_kernels.h
	CUDA_HDR += solvers/cuda_heunllg.h
	CUDA_HDR += solvers/cuda_heunllg_kernel.h
	CUDA_HDR += solvers/cuda_constrainedmc.h
	CUDA_HDR += thermostats/cuda_langevin_coth.h
	CUDA_HDR += thermostats/cuda_langevin_coth_kernel.h
	CUDA_HDR += thermostats/cuda_langevin_white.h

endif

ifeq ($(SYSTYPE),Darwin)
	CC = clang++
	CFLAGS += -fslp-vectorize
	BASIC_LDFLAGS += -framework Accelerate -Wl -rpath /usr/local/cuda/lib
else
	BASIC_LDFLAGS +=
endif


ifdef LIBCONFIGDIR
	BASIC_CFLAGS += -I$(LIBCONFIGDIR)/include
	BASIC_LDFLAGS += -L$(LIBCONFIGDIR)/lib
endif

EXTLIBS += -lconfig++

ifdef MKLROOT
	CC = icpc
	BASIC_CFLAGS += -I$(MKLROOT)/include -I$(MKLROOT)/include/fftw -D__INTEL_COMPILER -DMKL
	BASIC_LDFLAGS += -L$(MKLROOT)/lib/intel64
	EXTLIBS += -lmkl_intel_lp64 -lmkl_core -lmkl_sequential -lpthread -lm
else
	EXTLIBS += -lfftw3
endif

ifndef NO_CUDA
	BASIC_CFLAGS += -I$(CUDADIR)/include -DCUDA
	BASIC_CUFLAGS += -I$(CUDADIR)/include -DCUDA
	ifeq ($(SYSTYPE),Darwin)
		BASIC_LDFLAGS += -L$(CUDADIR)/lib
		BASIC_CUFLAGS += -std=c++11 -ccbin=/usr/bin/clang++ -Xcompiler "-DNDEBUG -march=native -O3 -g -funroll-loops" -Xlinker
	else
		BASIC_LDFLAGS += -L$(CUDADIR)/lib64
		BASIC_CUFLAGS += -std=c++11 -ccbin=/usr/bin/g++ -Xcompiler "-fno-finite-math-only -O3 -g -funroll-loops -DNDEBUG"
	endif
	EXTLIBS += -lcudart -lcurand -lcublas -lcusparse -lcufft
	ifdef CUDA_BUILD_FERMI
		BASIC_CUFLAGS += -gencode=arch=compute_20,code=sm_20
	endif
	ifdef CUDA_BUILD_KEPLAR
		BASIC_CUFLAGS += -gencode=arch=compute_30,code=sm_30
	endif
	ifdef CUDA_BUILD_MAXWELL
		BASIC_CUFLAGS += -gencode=arch=compute_50,code=sm_50 \
						 -gencode=arch=compute_52,code=sm_52 \
						 -gencode=arch=compute_53,code=sm_53
	endif
endif

ifdef H5DIR
	BASIC_CFLAGS += -I$(H5DIR)/include
	BASIC_LDFLAGS += -L$(H5DIR)/lib
endif
EXTLIBS += -lhdf5 -lhdf5_cpp -lsymspg -lblas

ALL_CFLAGS += $(BASIC_CFLAGS)
ALL_CUFLAGS += $(BASIC_CFLAGS) $(BASIC_CUFLAGS)
ALL_LDFLAGS += $(BASIC_LDFLAGS)

LIBS = $(EXTLIBS)

#-----------------------------------------------------------------
# Build Rules
#-----------------------------------------------------------------

.PHONY: all clean release devel debug FORCE

all:: jams++

jams++: $(OBJS) $(CUDA_OBJS)
	$(QUIET_LINK)$(CC) $(ALL_CFLAGS) -o $@ $(OBJS) $(CUDA_OBJS) $(ALL_LDFLAGS) $(LIBS)
	@echo
	@echo " JAMS++ build complete. "
	@echo
	@echo " System       ... $(SYSTYPE) "
	@echo " Architecture ... $(CPUTYPE) "
ifndef NO_CUDA
	@echo "              ... CUDA enabled"
endif
	@echo " Compiler     ... $(CC) "
	@echo

# core/jams++.o solvers: EXTRA_CPPFLAGS += \
# 	'-DGITCOMMIT="$(GITCOMMIT)"'

$(OBJS): %.o: %.cc $(HDR)
	$(QUIET_CC)$(CC) -o $*.o -c $(ALL_CFLAGS) $(EXTRA_CPPFLAGS) $<

$(CUDA_OBJS): %.o: %.cu $(HDR)
	$(QUIET_NVCC)$(NVCC) $(ALL_CUFLAGS) -o $*.o -c $<

install: jams++
	cp jams++ ~/local/bin/jams++-$(GITSHORT)
	ln -sf ~/local/bin/jams++-$(GITSHORT) ~/local/bin/jams++-unstable

clean:
	$(RM) core/*.o physics/*.o monitors/*.o solvers/*.o hamiltonian/*.o thermostats/*.o
	$(RM) jams++
