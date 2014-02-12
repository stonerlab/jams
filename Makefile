TOPDIR = .

include ./Makefile.in

source-dirs := core/ monitors/ physics/ solvers/

jams++ :: objects
	$(LD) -o $@ $(CFLAGS) $(LDFLAGS) $(LIBS) $(foreach d, $(source-dirs), $(wildcard $d*.o))

	@echo
	@echo " JAMS++ build complete. "
	@echo
	@echo " System       ... $(systype) "
	@echo " Architecture ... $(cputype) "
ifeq ($(withcuda), 1)
	@echo "              ... CUDA enabled"
endif
	@echo " Compiler     ... $(CXX)     "
ifeq ($(withdebug), 1)
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
