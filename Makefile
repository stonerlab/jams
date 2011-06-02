TOPDIR = .

include ./Makefile.in

#SUBDIRS = src src/monitors src/solvers

source-dirs := $(sort $(dir $(shell find . -name '*.cc')))

cuda-kernels := src/solvers/cuda_semillg

amorphus:: objects
	$(CXX) $(CFLAGS) -c misc/amorphus.cpp -o misc/amorphus.o
	$(LD) -o $@ $(CFLAGS) $(LDFLAGS) misc/amorphus.o $(foreach d, $(source-dirs), $(wildcard $d*.o)) $(LIBS) 

jams++ :: objects kernels
	rm -rf misc/amorphus.o
	$(LD) -o $@ $(CFLAGS) $(LDFLAGS) $(foreach d, $(source-dirs), $(wildcard $d*.o)) $(LIBS) 
	

	@echo
	@echo " JAMS++ build complete. "
	@echo
	@echo " System       ... $(systype) "
	@echo " Architecture ... $(cputype) "
ifeq ($(withcuda),1)
	@echo "              ... CUDA enabled"
endif
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
		  $(MAKE) -C $$d $(@F) || exit 1; \
		fi; \
	done

kernels : $(cuda-kernels).o
	for d in $(cuda-kernels); do  \
		$(CUDA) -arch sm_20 -O3 $(INCLUDES) --maxrregcount=32 --ptxas-options=-v -c $${d}.cu -o $${d}.o; \
	done

clean :
	rm -f jams++
	for d in $(source-dirs) ; \
		do if test -d $$d; then \
		  $(MAKE) -C $$d $(@F) || exit 1; \
		fi; \
	done
