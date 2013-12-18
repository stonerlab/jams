TOPDIR = ..

include $(TOPDIR)/Makefile.in

OBJS = $(patsubst %.cc, %.o, $(wildcard *.cc))
CUDAOBJS = $(patsubst %.cu, %.o, $(wildcard *.cu))

ifeq ($(withcuda),1)
objects: $(OBJS) $(CUDAOBJS) $(HEADERS)
endif

ifeq ($(withcuda),0)
objects: $(OBJS) $(HEADERS)
endif

clean:
	rm -f *.o

$(OBJS) : $(HEADERS) $(TOPDIR)/Makefile.in Makefile

%.o : %.cc
	$(CXX) $(CFLAGS) $(SOURCEFILE) $(OUTPUTFILE)

%.o : %.cu
	$(CUDA) $(INCLUDES) $(CUDAOPTS) $(SOURCEFILE) $(OUTPUTFILE)
