TOPDIR = .

include ./Makefile.in

#SUBDIRS = src src/monitors src/solvers

source-dirs := $(sort $(dir $(shell find . -name '*.cc')))

cuda-kernels := src/solvers/cuda_semillg

jams++ :: objects kernels
	$(LD) -o $@ $(CFLAGS) $(LDFLAGS) $(foreach d, $(source-dirs), $(wildcard $d*.o)) $(LIBS) 
	

	@echo
	@echo " JAMS++ build complete. "
	@echo
	@echo " System       ... $(systype) "
	@echo " Architecture ... $(cputype) "
	@echo " Compiler     ... $(CXX)     "
ifeq ($(withdebug),1)
		@echo " Build type   ... Debug      "
else
		@echo " Build type   ... Production "
endif
	@echo

objects :
	for d in $(source-dirs) ; \
		do if test -d $$d; then \
		  $(MAKE) -j4 -C $$d $(@F) || exit 1; \
		fi; \
	done

kernels :
	for d in $(cuda-kernels); do  \
		nvcc -arch sm_13 -O3 $(INCLUDES) --maxrregcount=32 --ptxas-options=-v -c $${d}.cu -o $${d}.o; \
	done

clean :
	rm -f jams++
	for d in $(source-dirs) ; \
		do if test -d $$d; then \
		  $(MAKE) -j4 -C $$d $(@F) || exit 1; \
		fi; \
	done
