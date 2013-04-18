TOPDIR = ../..

include $(TOPDIR)/Makefile.in

OBJS = $(patsubst %.cc, %.o, $(wildcard *.cc))
CUDAOBJS = $(patsubst %.cu, %.o, $(wildcard *.cu))
LOCALHEAD = $(wildcard *.h)

ifeq ($(withcuda),1)
objects: $(OBJS) $(CUDAOBJS) $(HEADERS) $(LOCALHEAD)
endif

ifeq ($(withcuda),0)
objects: $(OBJS) $(HEADERS) $(LOCALHEAD)
endif

clean:
	rm -f *.o

$(OBJS) : $(HEADERS) $(TOPDIR)/Makefile.in Makefile

%.o : %.cc $(LOCALHEAD)
	$(CXX) $(CFLAGS) $(SOURCEFILE) $(OUTPUTFILE)

%.o : %.cu $(LOCALHEAD)
	$(CUDA) $(INCLUDES) $(CUDAOPTS) $(SOURCEFILE) $(OUTPUTFILE)
