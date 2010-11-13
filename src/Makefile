TOPDIR = ..

include $(TOPDIR)/Makefile.in

OBJS = $(patsubst %.cc, %.o, $(wildcard *.cc))

objects: $(OBJS) $(HEADERS)

clean:
	rm -f $(OBJS)

$(OBJS) : $(HEADERS) $(TOPDIR)/Makefile.in Makefile

%.o : %.cc
	$(CXX) $(CFLAGS) $(SOURCEFILE) $(OUTPUTFILE)
