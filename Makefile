TOPDIR = .

include ./Makefile.in

source-dirs := $(sort $(dir $(shell find . -name '*.cc' -o -name '*.cu')))

jams++ :: objects
	$(LD) -o $@ $(CFLAGS) $(LDFLAGS) $(foreach d, $(source-dirs), $(wildcard $d*.o)) $(LIBS) 

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

clean :
	rm -f jams++
	for d in $(source-dirs) ; \
		do if test -d $$d; then \
		  $(MAKE) -C $$d $(@F) || exit 1; \
		fi; \
	done 
