TOPDIR = ../..

include $(TOPDIR)/Makefile.in

OBJS = $(patsubst %.cc, %.o, $(wildcard *.cc))
CUDAOBJS = $(patsubst %.cu, %.o, $(wildcard *.cu))

objects: $(OBJS) $(HEADERS)

clean:
	rm -f *.o

$(OBJS) : $(HEADERS) $(TOPDIR)/Makefile.in Makefile

%.o : %.cc
	$(CXX) $(CFLAGS) $(SOURCEFILE) $(OUTPUTFILE)
