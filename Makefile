TOPDIR = .

include ./Makefile.in

source-dirs := $(sort $(dir $(shell find . -name '*.cc')))

cuda-kernels := src/solvers/cuda_semillg src/solvers/cuda_heunllg

ifeq ($(withcuda),1)
jams++ :: objects kernels
	$(LD) -o $@ $(CFLAGS) $(LDFLAGS) $(foreach d, $(source-dirs), $(wildcard $d*.o)) $(LIBS) 
endif

ifeq ($(withcuda),0)
jams++ :: objects
	$(LD) -o $@ $(CFLAGS) $(LDFLAGS) $(foreach d, $(source-dirs), $(wildcard $d*.o)) $(LIBS) 
endif

ifeq ($(systype),Darwin)
	$(warning Disabling CUDA on Darwin platform (64bit incompatible))
endif
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

kernels : $(foreach d, $(cuda-kernels), $(wildcard $d*.o)) 
	for d in $(cuda-kernels); do  \
		$(CUDA) \
			-gencode=arch=compute_20,code=sm_20 -gencode=arch=compute_20,code=compute_20 \
			-gencode=arch=compute_13,code=sm_13 -gencode=arch=compute_13,code=compute_13 \
			-O3 -DNDEBUG -DCUDA $(INCLUDES) --ptxas-options=-v -c $${d}.cu -o $${d}.o; \
	done

clean :
	rm -f jams++
	for d in $(source-dirs) ; \
		do if test -d $$d; then \
		  $(MAKE) -C $$d $(@F) || exit 1; \
		fi; \
	done 
