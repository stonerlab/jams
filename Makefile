TOPDIR = .

include ./Makefile.in

#SUBDIRS = src src/monitors src/solvers

source-dirs := $(sort $(dir $(shell find . -name '*.cc')))

jams++ :: objects
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
